import comet_ml
import argparse
import importlib
import os
import time
import traceback
import yaml
from common_py_utils.common_utils import SimpleNamespace
from common_py_utils import yaml_cfg_parser
import tensorflow.compat.v1 as tf
import concurrent.futures


def arg_parse():
    parser = argparse.ArgumentParser(description='Model training function, loads a model and dataset and runs training.')

    parser.add_argument(
        '-p', '--params', action='append',
        help='YAML params file(s), specify 1 or more YAML files containing the necessary hyperparameters, '
             'training parameters, and dataset parameters. The files may be local or on S3. '
             'To specify multiple files use multiple --params options for each file.'
    )

    parser.add_argument(
        '-o', '--override', action='append',
        help='Overrides value(s) in the parameters yaml files that are parsed with --params. All valid one line yaml'
             'accepted. Use https://onlineyamltools.com/minify-yaml to convert yaml to one line. This parameter'
             'can be specified multiple times.'
             'Example: {trainingparams: {training_steps: 20, checkpoint: ../checkpoints/dev/SCF05/}}'
    )

    parsed_args = vars(parser.parse_args())
    return parsed_args


def train(trainingparams: dict, modelparams: dict, datasetparams: dict):
    print('trainingparams', yaml.dump(trainingparams), sep='\n')
    print('modelparams', yaml.dump(modelparams), sep='\n')
    print('datasetparams', yaml.dump(datasetparams), sep='\n')

    training_steps = trainingparams['training_steps']
    dataset_class = trainingparams['dataset_class']
    model_class = trainingparams['model_class']
    load_model = trainingparams.get('load_model', None)
    checkpoint = trainingparams['checkpoint']
    testeval_on_checkpoint = trainingparams['testeval_on_checkpoint']
    checkpoint_final = trainingparams.get('checkpoint_final', None)
    debug = trainingparams.get('debug', False)

    dataset_module = importlib.import_module('.'.join(dataset_class.split('.')[:-1]))
    dataset = getattr(dataset_module, dataset_class.split('.')[-1])(**datasetparams)

    model_module = importlib.import_module('.'.join(model_class.split('.')[:-1]))
    model = getattr(model_module, model_class.split('.')[-1])(**modelparams, dataset_obj=dataset)

    ds = dataset.as_dataset()
    get_next_tensor = tf.data.make_one_shot_iterator(ds).get_next()

    model.build_model(input_tensors=get_next_tensor)
    model.init()

    if load_model is not None:
        model.load_model(load_model)

    # For comet output logging and progress bar issues try: 'simple', 'native', or None
    experiment = comet_ml.Experiment(
        api_key='uCZTtx2dLDpXYFu0aXzmgygn6',
        project_name='sleep-state-model',
        disabled=trainingparams['disable_comet'],
        auto_output_logging='simple'
    )
    experiment.set_name(os.getenv('HOSTNAME')[16:])
    experiment.set_model_graph(tf.get_default_graph())
    executor_post_processing = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    experiment.log_parameters({'trainingparams.' + k: v for k, v in trainingparams.items()})
    experiment.log_parameters({'modelparams.' + k: v for k, v in modelparams.items()})
    experiment.log_parameters({'datasetparams.' + k: v for k, v in datasetparams.items()})

    # Sanity check that we don't leave debug settings accidentally on
    if training_steps < 1000:
        print('*' * 50 + '\n' + '**\n' * 5 + '** WARNING - TRAINING STEPS BELOW 1000\n' + '** \n' * 5 + '*' * 50 + '\n')

    # Set initial global step (will be non-zero if an existing model was loaded)
    out = SimpleNamespace(**model.session.run(fetches={'global_step': model.ops.global_step}))
    last_step_time = time.time()  # used to compute the time per training step for printing to stdout

    while out.global_step < training_steps:
        fetch_ops = {
            f.name: f.tensor for f in model.fetch_ops
            if (out.global_step + 1) % f.update_frequency == 0
        }

        # Run train op, out is a SimpleNamespace object which supports dot notation access
        out = SimpleNamespace(
            **model.session.run(fetches=fetch_ops),
            global_step=model.session.run(model.ops.global_step)
        )

        # Comet.ml metrics
        for op in [op for op in model.fetch_ops if op.is_reported and hasattr(out, op.name)]:
            if debug:
                print(f'DEBUG> logging metric {op.name} with value {out[op.name]} at step {out.global_step}')
            experiment.log_metric(op.name, out[op.name], step=out.global_step)

        # Comet.ml optional images & text - processing in a thread in parallel to the next step
        if hasattr(model, 'log_text'):
            executor_post_processing.submit(log_texts, model, experiment, out)
        if hasattr(model, 'log_image'):
            executor_post_processing.submit(log_images, model, experiment, out)
        if hasattr(model, 'update_callback'):
            executor_post_processing.submit(update_callback, model, out)

        extra_status = model.status_message(out)
        extra_status = '' if extra_status is None else ' | ' + extra_status
        step_time = time.time() - last_step_time
        last_step_time = time.time()
        status = f'{out.global_step:d}/{training_steps:d} | {step_time:.2f}s{extra_status}'
        print(status)

        # Save periodic checkpoints and start test eval job when appropriate
        chkpt_file = model.save_checkpoint_periodically(checkpoint, out.global_step, out.loss, force=False)
        if chkpt_file is not None and testeval_on_checkpoint:
            submit_test_eval_job(chkpt_file, ','.join(test_video_files), experiment.get_key(), include_modules)  # todo change this to pass the YAML not "include_modules"

    # Save final checkpoint
    chkpt_file = model.save_checkpoint_periodically(
        checkpoint_final if checkpoint_final is not None else checkpoint, out.global_step, out.loss, force=True
    )
    if testeval_on_checkpoint:
        submit_test_eval_job(chkpt_file, ','.join(test_video_files), experiment.get_key(), include_modules)

    experiment.end()
    print('Comet.ml experiment ended successfully, dataset shutdown begin.')
    dataset.shutdown()
    print('Dataset shutdown successful.')


def log_texts(model, experiment, out):
    try:
        for text in model.log_text(out) or []:
            experiment.log_text(text, out.global_step)
    except Exception as e:
        traceback.print_exc()
        raise e


def log_images(model, experiment, out):
    try:
        for name, image in model.log_image(out) or []:
            experiment.log_image(image, name=name, step=out.global_step)
    except Exception as e:
        traceback.print_exc()
        raise e


def update_callback(model, out):
    try:
        model.update_callback(out)
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    args = arg_parse()
    params = yaml_cfg_parser.parse_yaml_cfg(args['params'], is_file=True, includes=args['override'])
    train(params['trainingparams'], params['modelparams'], params['datasetparams'])
