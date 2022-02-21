import numpy as np
import csv
import io
import argparse
from common_py_utils import yaml_cfg_parser
from braingeneers.utils import smart_open
import zipfile
import math
import sklearn
import sklearn.metrics


def arg_parser():
    parser_stats = argparse.ArgumentParser(description='Sleepstate summary statistics.')

    parser_stats.add_argument(
        '--predictions_filename', type=str, required=True,
        help='Input path/filename to the predictions_DATASET_NAME.csv, maybe be local or S3.'
    )
    parser_stats.add_argument(
        '--output_filename', type=str, required=False, default=None,
        help='Path/filename of output file, defaults to path/summary_stats_DATASET_NAME.csv.'
    )
    parser_stats.add_argument(
        '--quiet', action='store_false', required=False, default=True, dest='display',
        help='Flag, do not display results to STDOUT (defaults to writing to STDOUT and file.'
    )
    parser_stats.add_argument(
        '--test_video_files', type=str, required=False,
        help='Comma separated list of video file names that were held out as test and will be marked as such.'
    )
    parser_stats.add_argument(
        '-p', '--params', action='append',
        help='YAML params file(s), specify 1 or more YAML files containing the necessary dataset parameters. '
             'The files may be local or on S3. To specify multiple files use multiple --params options for each file.'
    )
    parser_stats.add_argument(
        '-o', '--override', action='append',
        help='Overrides value(s) in the parameters yaml files that are parsed with --params. All valid one line yaml'
             'accepted. Use https://onlineyamltools.com/minify-yaml to convert yaml to one line. This parameter'
             'can be specified multiple times.'
             'Example: {trainingparams: {training_steps: 20, checkpoint: ../checkpoints/dev/SCF05/}}'
    )

    return parser_stats.parse_args()


# noinspection DuplicatedCode
def generate_summary_statistics(predictions_filename: (str, io.IOBase), output_filename: (str, io.IOBase),
                                test_video_files: (str, list, tuple) = None, display: bool = True):
    """
    Run from command line, produces a set of summary statistics written to output_file

    :param predictions_filename: string (local file) or file-like object, containing predictions CSV.
    :param output_filename: default to None to use the same DATASET_NAME as parsed from predictions_file for output, or
           path/filename.txt of output file to generate.
    :param display: True/False to display results to stdout as well as write to file.
    :param test_video_files: a comma separated list of video filenames held out as test set, formatted as per dataset_config standard
    :return: total-accuracy, (per-label-accuracy, ...), (per-video-filename-accuracy, ...)
    """
    output = []

    test_video_files = \
        test_video_files if isinstance(test_video_files, (list, tuple)) else\
        () if test_video_files is None else \
        test_video_files.split(',') if isinstance(test_video_files, str) else None
    assert test_video_files is not None, f'Invalid value for test_video_files: {test_video_files}'

    predictions_bytes = read_file_bytes(predictions_filename)
    predictions = np.recfromtxt(io.BytesIO(predictions_bytes), names=True, delimiter=',')

    # Threshold set at half HP_MAX_PREDICT_TIME_FPS, which defines what "close to a state change" means
    masks = [
        ('All Samples', np.ones(predictions.shape[0], dtype=np.bool)),
        ('Between State Change', np.abs(predictions['label_time']) >= 0.5),
        ('Near State Change', np.abs(predictions['label_time']) < 0.5),
    ]

    for title_text, mask_time in masks:
        # test-set
        mask_test_set = np.isin(predictions['video_filename'].astype(np.str), test_video_files)
        predictions_filtered = predictions[mask_test_set & mask_time]
        count_wake, count_nrem, count_rem, acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time, f1micro_wnr, f1macro_wnr, ba_wnr = \
            summary_statistics(
                label_sleepstate=predictions_filtered['label_wnr_012'],
                label_time=predictions_filtered['label_time'],
                predicted_wnr=predictions_filtered['predicted_wnr_012'],
                predicted_ma=None,  # predictions_filtered['predicted_ma_01'],
                predicted_pa=None,  # predictions_filtered['predicted_pa_01'],
                predicted_time=None,  # predictions_filtered['predicted_time'],
            )
        output += [f'per-label-test-set ({title_text}) ({count_wake + count_nrem + count_rem} samples)']
        output += _formatter_dataset(count_wake, count_nrem, count_rem, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, f1micro_wnr, f1macro_wnr, ba_wnr)
        output += ['']
    
        # train-set
        mask_train_set = ~np.isin(predictions['video_filename'].astype(np.str), test_video_files)
        predictions_filtered = predictions[mask_train_set & mask_time]
        count_wake, count_nrem, count_rem, acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time, f1micro_wnr, f1macro_wnr, ba_wnr = \
            summary_statistics(
                label_sleepstate=predictions_filtered['label_wnr_012'],
                label_time=predictions_filtered['label_time'],
                predicted_wnr=predictions_filtered['predicted_wnr_012'],
                predicted_ma=None,  # predictions_filtered['predicted_ma_01'],
                predicted_pa=None,  # predictions_filtered['predicted_pa_01'],
                predicted_time=None,  # predictions_filtered['predicted_time'],
            )
        output += [f'per-label-train-set ({title_text}) ({count_wake + count_nrem + count_rem} samples)']
        output += _formatter_dataset(count_wake, count_nrem, count_rem, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, f1micro_wnr, f1macro_wnr, ba_wnr)
        output += ['']
    
        # per-video-filename-accuracy
        output += [f'per-video-metrics ({title_text})']
        output += ['                                   Passive/   Micro-']
        output += ['    WAKE   |   NREM   |   REM    |  Active  |  arousal | video-filename']
        output += ['  -------- | -------- | -------- | -------- | -------- | ---------------']
        for video in np.unique(predictions['video_filename']):
            video = video.decode()
            is_test_video = True if np.isin(video, test_video_files) else False
            mask_video = predictions['video_filename'].astype(np.str) == video
            predictions_filtered = predictions[mask_video & mask_time]

            if len(predictions_filtered) > 0:
                count_wake, count_nrem, count_rem, acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time, f1micro_wnr, f1macro_wnr, ba_wnr = \
                    summary_statistics(
                        label_sleepstate=predictions_filtered['label_wnr_012'],
                        label_time=predictions_filtered['label_time'],
                        predicted_wnr=predictions_filtered['predicted_wnr_012'],
                        predicted_ma=None,  # predictions_filtered['predicted_ma_01'],
                        predicted_pa=None,  # predictions_filtered['predicted_pa_01'],
                        predicted_time=None,  # predictions_filtered['predicted_time'],
                    )
            else:
                acc_wake = acc_nrem = acc_rem = acc_pa = f1_ma = math.nan
    
            output += ['    {:.2f}   |   {:.2f}   |   {:.2f}   |   {:.2f}   |   {:.2f}   | {}{}'
                           .format(acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, video,
                                   ' (test-set)' if is_test_video else '').replace('nan', ' NaN')]
        output += ['', '']

    if display:
        print()
        for line in output:
            print(line)
        print()

    # Write to file
    if output_filename is None:
        dataset_name = predictions_filename.split('_')[1].split('.')[0]
        basepath = os.path.split(predictions_filename)[0]
        output_file_open = smart_open.open(os.path.join(basepath, 'summary_statistics_{}.txt'.format(dataset_name)), 'w+')
    elif isinstance(output_filename, str):
        output_file_open = smart_open.open(output_filename, 'w')
    else:
        output_file_open = output_filename
    for o in output:
        output_file_open.write('{}\n'.format(o))
    output_file_open.close()

    return output


# noinspection DuplicatedCode,PyUnusedLocal
def summary_statistics(label_sleepstate, label_time,
                       predicted_wnr,
                       predicted_ma, predicted_pa, predicted_time):
    """ Generates raw numbers for summary statistics for a set of predictions """

    # mask_wake = np.isin(label_sleepstate, [1, 5])
    # mask_nrem = label_sleepstate == 2
    # mask_rem = label_sleepstate == 3
    mask_wake = label_sleepstate == 0
    mask_nrem = label_sleepstate == 1
    mask_rem = label_sleepstate == 2

    # mask_time = np.abs(label_time) <= 1.0

    count_wake = np.sum(mask_wake)
    count_nrem = np.sum(mask_nrem)
    count_rem = np.sum(mask_rem)

    acc_wake = np.sum(predicted_wnr[mask_wake] == 0) / count_wake
    acc_nrem = np.sum(predicted_wnr[mask_nrem] == 1) / count_nrem
    acc_rem = np.sum(predicted_wnr[mask_rem] == 2) / count_rem
    # acc_pa = np.sum((((predicted_pa[mask_wake] - 1) * -4) + 1) == label_sleepstate[mask_wake]) / np.sum(mask_wake) \
    #     if all(predicted_pa >= 0) else math.nan
    # f1_ma = sklearn.metrics.f1_score(
    #     y_true=(label_sleepstate == 4).astype(predicted_ma.dtype),
    #     y_pred=predicted_ma
    # ) if len(predicted_ma) > 0 and all(predicted_ma >= 0) else math.nan
    # rmse_time = np.sqrt(((label_time[mask_time] - predicted_time[mask_time])**2).mean())
    f1micro_wnr = sklearn.metrics.f1_score(y_true=label_sleepstate, y_pred=predicted_wnr, average='micro')
    f1macro_wnr = sklearn.metrics.f1_score(y_true=label_sleepstate, y_pred=predicted_wnr, average='macro')
    ba_wnr = sklearn.metrics.balanced_accuracy_score(y_true=label_sleepstate, y_pred=predicted_wnr)

    n_wnr = np.sum(mask_wake ^ mask_nrem ^ mask_rem)
    acc_wnr = acc_wake * (np.sum(mask_wake) / n_wnr) + \
                acc_nrem * (np.sum(mask_nrem) / n_wnr) + \
                acc_rem * (np.sum(mask_rem) / n_wnr)

    return count_wake, count_nrem, count_rem, acc_wnr, acc_wake, acc_nrem, acc_rem, np.nan, np.nan, np.nan, f1micro_wnr, f1macro_wnr, ba_wnr  # NAN's were: acc_pa, f1_ma, rmse_time


def read_file_bytes(filename):
    """ Reads a file bytes, the file may be local or S3 and may be zipped (single file) or not. Bytes returned."""
    print('Getting file: {}'.format(filename))
    bytes_raw = smart_open.open(filename, 'rb').read()
    if filename.endswith('.zip'):
        z = zipfile.ZipFile(io.BytesIO(bytes_raw))
        bytes_raw = z.read(z.infolist()[0])
    return bytes_raw


# noinspection PyUnusedLocal
def _formatter_dataset(count_wake, count_nrem, count_rem, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, f1micro_wnr, f1macro_wnr, ba_wnr):
    output = []
    output += [f'   {   acc_wake:.2f}: WAKE            (accuracy) ({count_wake: >7} samples)'.replace('nan', ' NaN')]
    output += [f'   {   acc_nrem:.2f}: NREM            (accuracy) ({count_nrem: >7} samples)'.replace('nan', ' NaN')]
    output += [f'   {    acc_rem:.2f}: REM             (accuracy) ({ count_rem: >7} samples)'.replace('nan', ' NaN')]
    output += [f'   {f1micro_wnr:.2f}: F1 Multiclass  (avg=micro) ({count_wake + count_nrem + count_rem: >7} samples)'.replace('nan', ' NaN')]
    output += [f'   {f1macro_wnr:.2f}: F1 Multiclass  (avg=macro) ({count_wake + count_nrem + count_rem: >7} samples)'.replace('nan', ' NaN')]
    output += [f'   {     ba_wnr:.2f}: Balanced Accuracy Score    ({count_wake + count_nrem + count_rem: >7} samples)'.replace('nan', ' NaN')]
    # output += ['   {:.2f}: Passive/Active  (accuracy)'.format(acc_pa).replace('nan', ' NaN')]
    # output += ['   {:.2f}: Micro-arousal   (F1 score)'.format(f1_ma).replace('nan', ' NaN')]
    return output


if __name__ == '__main__':
    args = vars(arg_parser())
    datasetparams = yaml_cfg_parser.parse_yaml_cfg(args['params'], is_file=True, includes=args['override'])['datasetparams']
    generate_summary_statistics(predictions_filename=args['predictions_filename'],
                                output_filename=args['output_filename'],
                                test_video_files=datasetparams['test_video_files'],
                                display=args['display'])
