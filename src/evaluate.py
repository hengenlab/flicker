import argparse
from common_py_utils import yaml_cfg_parser
import importlib
import tensorflow.compat.v1 as tf
import concurrent.futures
import yaml
import time
import zipfile
import os.path


def arg_parse():
    parser = argparse.ArgumentParser(description='Model evaluation function, loads a model and dataset and runs training.')

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
             'can be specified multiple times. '
             'Example: {trainingparams: {training_steps: 20, checkpoint: ../checkpoints/dev/SCF05/}}'
    )

    parsed_args = vars(parser.parse_args())
    return parsed_args


def evaluate(evaluateparams: dict, modelparams: dict, datasetparams: dict):
    """ Evaluates a dataset on a single device and saves the results to output_file in evaluateparams
    :param evaluateparams: Yaml configuration for evaluate method
    :param modelparams: Yaml configuration for the model
    :param datasetparams: Yaml configuration for the dataset
    :return:
    """
    print('evaluateparams', yaml.dump(evaluateparams), sep='\n')
    print('modelparams', yaml.dump(modelparams), sep='\n')
    print('datasetparams', yaml.dump(datasetparams), sep='\n')

    dataset_class = evaluateparams['dataset_class']
    model_class = evaluateparams['model_class']
    load_model = evaluateparams['load_model']
    limit = evaluateparams.get('limit', None)

    dataset_module = importlib.import_module('.'.join(dataset_class.split('.')[:-1]))
    dataset = getattr(dataset_module, dataset_class.split('.')[-1])(**datasetparams)

    model_module = importlib.import_module('.'.join(model_class.split('.')[:-1]))
    model = getattr(model_module, model_class.split('.')[-1])(**modelparams, dataset_obj=dataset)

    # Build dataset
    ds = dataset.as_dataset()
    get_next_tensor = tf.data.make_one_shot_iterator(ds).get_next()

    # Build model
    model.build_model(input_tensors=get_next_tensor)
    model.init()
    model.load_model(load_model)

    sess = model.session

    executor_save_output = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future_async_save_output = concurrent.futures.Future()
    future_async_save_output.set_result(None)

    fetch_ops = {f.name: f.tensor for f in model.fetch_ops_evaluate}
    processed_count = 0
    output_filename_tmp = '/tmp/output_file.csv'
    output_filename = evaluateparams['output_file']  # + ('' if evaluateparams['output_file'].endswith('.zip') else '.zip')
    t0 = time.time()

    # Compute predictions and save the results (asynchronously) to disk
    with sess, open(output_filename_tmp, 'w') as output_file:
        try:
            # An expected OutOfRangeError exception breaks out of the loop
            while limit is None or processed_count < limit:
                tf_output = sess.run(fetches=fetch_ops)
                processed_count += list(tf_output.values())[0].shape[0]
                print(f'Processed {processed_count}/{len(dataset.labels_matrix)} samples in {(time.time() - t0):.2f} seconds.')
                t0 = time.time()

                # Check the previous write has completed, if not warn the user
                if not future_async_save_output.done():
                    t00 = time.time()
                    print('Warning, waiting on save_output to complete... ', end='')
                    future_async_save_output.result()
                    print('Delay of {} seconds occurred.'.format(time.time() - t00))
                future_async_save_output.result()  # result is None, but we get result in case of exception

                # Asynchronously save the last output to disk
                future_async_save_output = executor_save_output.submit(
                    save_output, output_file=output_file, tf_output=tf_output, model=model,
                )

        except tf.errors.OutOfRangeError:
            pass  # normal occurrence meaning the dataset has been exhausted

        # wait for the last async write to complete
        future_async_save_output.result()

        # Zip and copy csv file to final destination
        with zipfile.ZipFile(output_filename_tmp + '.zip', mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(output_filename_tmp, arcname=os.path.basename(output_filename))
        tf.io.gfile.copy(output_filename_tmp + '.zip', output_filename + '.zip', overwrite=True)

        print(f'Saved results to: {output_filename}')


def save_output(output_file, tf_output: dict, model):
    """ Streams the output from each update step to disk. """
    fetch_ops_evaluate = model.fetch_ops_evaluate
    batch_size = tf_output[fetch_ops_evaluate[0].name].shape[0]

    # CSV header
    if output_file.tell() == 0:
        csv_header_as_list = [f.name for f in fetch_ops_evaluate]
        output_file.write(','.join(csv_header_as_list) + '\n')

        # assert all batch dimensions are the same shape
        for f in fetch_ops_evaluate:
            assert tf_output[f.name].shape[0] == batch_size, \
                f'All batch dimensions are expected to be the same shape, {f.name} has a ' \
                f'batch dimension of {f.tensor.shape[0]} but {batch_size} was expected.'

    # Write a CSV line for each sample in the batch, this assumes dimension 0 is the batch dimension.
    for i in range(batch_size):
        csv_row_as_list = [to_string(tf_output[f.name][i]) for f in fetch_ops_evaluate]
        output_file.write(','.join(csv_row_as_list) + '\n')


def to_string(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    else:
        return str(val)


if __name__ == '__main__':
    args = arg_parse()
    params = yaml_cfg_parser.parse_yaml_cfg(args['params'], is_file=True, includes=args['override'])
    evaluate(params['evaluateparams'], params['modelparams'], params['datasetparams'])
