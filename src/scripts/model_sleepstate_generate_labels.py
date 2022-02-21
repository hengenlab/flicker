""" Simple functions for configuring the neural labels dataset CSV. Originally used for the MVP. """
import neuraltoolkit as ntk
import numpy as np
import itertools
import os
import argparse
import fs as pyfilesystem
from fs_s3fs import S3FS
# import dataset
import tensorflow.compat.v1 as tf
import uuid
from common_py_utils import yaml_cfg_parser


def parse_args():
    parser = argparse.ArgumentParser(description='Script for generating sleep state model labels.')

    parser.add_argument(
        '--output', type=str, required=False, default='labels_sleepstate',
        help='The output path/filename (without extension), a CSV and NPZ file will be created with this name.'
    )
    parser.add_argument(
        '--neural_files_basepath', nargs='?', type=str,
        help='neural files directory path or S3.'
    )
    parser.add_argument(
        '--video_files_basepath', nargs='?', type=str,
        help='video files directory path or S3.'
    )
    parser.add_argument(
        '--sleepstate_files_basepath', nargs='?', type=str,
        help='sleep state files directory path or S3.'
    )
    parser.add_argument(
        '--syncpulse_files_zip', nargs='?', type=str,
        help='syncpulse files .zip local or S3.'
    )
    parser.add_argument(
        '--neural_bin_files_per_sleepstate_file', type=int, default=12,
        help='Number of neural binary files that are covered by each sleep state '
             'numpy file. For example in EAB40 there are 12 neural files per sleep state file, therefore the '
             'parameter is 12; in SCF05 there is only one sleep state file for ALL neural data files, in this case the '
             'parameter is -1 for ALL.'
    )
    parser.add_argument(
        '--manual_video_neural_offset_sec', type=int, required=False, default=0,
        help='See function documentation in function map_videoframes_to_syncpulse for details.'
    )
    parser.add_argument(
        '--initial_neural_file_for_sleepstate_mapping', type=int, default=0,
        help='If sleep state data doesn''t map to the first neural file present in the dataset, set this '
             'parameter to the number of neural files to skip before mapping to the sleep state data. '
             'For example, EAB50 maps sleep state to location 228 (0 indexed), which is file: '
             'Headstages_512_Channels_int16_2019-06-21_12-05-11.bin'
    )
    parser.add_argument(
        '--ignore_ecube_deviation_sec', type=float, default=0.00012,
        help='A deviation in ecube times indicates that the number of samples found in the .bin file '
             'differs from the ecube timestamp found in the beginning of the file compared to the timestamp '
             'of the next file. Ideally the number of samples will exactly match the ecube times '
             'reported across files. Small deviations of the default 0.00012 sec deviation equate to '
             'a maximum of 3 samples being dropped. Any deviation above this value will report a detail '
             'error and fail the process. Increase this value to ignore larger deviations, being aware '
             'that the deviation may represent a problem in the data and in mapping the neural '
             'data to video, labels, etc.'
    )
    parser.add_argument(
        '--n_channels', type=int, default=None,
        help='Number of channels in neural data, if unspecified the value will be inferred from the neural file names.'
    )
    parser.add_argument(
        '--fs', type=int, default=25000,
        help='Frame speed, 25000 = 15fps, default = 25000.'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False, help='Overwrite existing file.'
    )
    parser.add_argument(
        '--include_unlabeled', action='store_true', default=False,
        help='Flag to include samples with no sleep state label.'
    )
    parser.add_argument(
        '--dataset_yaml', required=False, type=str,
        help='A dataset .cfg file which defines specific details and quirks of the dataset. Any values specified '
             'in this file override the command line parameters.'
    )

    # v = dataset.parse_args_with_dataset_config(parser)

    args = vars(parser.parse_args())

    if 'dataset_yaml' in args:
        args_yaml = yaml_cfg_parser.parse_yaml_cfg(args['dataset_yaml'], is_file=True)
        for k, v in list(args_yaml['datasetparams'].items()) + list(args_yaml['datapreprocessingparams'].items()):
            if k in args:
                args[k] = v
        del args['dataset_yaml']

    return args


def main(output: str,
         neural_files_basepath: str,
         video_files_basepath: str,
         sleepstate_files_basepath: str,
         syncpulse_files_zip: str,
         neural_bin_files_per_sleepstate_file: int,
         manual_video_neural_offset_sec: int,
         n_channels: int, fs: int,
         initial_neural_file_for_sleepstate_mapping: int,
         overwrite: bool,
         ignore_ecube_deviation_sec: float,
         include_unlabeled: bool):

    # Sync files from S3 when appropriate
    neural_files = os.path.join(neural_files_basepath, '*.bin')

    if video_files_basepath.startswith('s3://'):
        os.system('aws --endpoint https://s3.nautilus.optiputer.net s3 sync '
                  '{} ../../tmp/gen_labels/Video/'.format(video_files_basepath))
        video_files = '../../tmp/gen_labels/Video/*.mp4'
    else:
        video_files = os.path.join(video_files_basepath, '*.mp4')

    if sleepstate_files_basepath.startswith('s3://'):
        os.system('aws --endpoint https://s3.nautilus.optiputer.net s3 sync '
                  '{} ../../tmp/gen_labels/SleepState/'.format(sleepstate_files_basepath))
        sleepstate_files = '../../tmp/gen_labels/SleepState/*.npy'
    else:
        sleepstate_files = os.path.join(sleepstate_files_basepath, '*.npy')

    if syncpulse_files_zip.startswith('s3://'):
        os.system('aws --endpoint https://s3.nautilus.optiputer.net s3 cp '
                  '{} ../../tmp/gen_labels/SyncPulse.zip'.format(syncpulse_files_zip))
        syncpulse_files_zip = '../../tmp/gen_labels/SyncPulse.zip'
    os.system('unzip -o -j {} -d ../../tmp/gen_labels/SyncPulse/'.format(syncpulse_files_zip))
    print('SyncPulse files unzipped successfully. Running NTK Sync...')
    syncpulse_files = '../../tmp/gen_labels/SyncPulse/*.bin'

    # Run NTK Sync
    output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files, \
        dlclabel_files, camera_pulse_output_matrix, pulse_ix = \
        ntk.map_video_to_neural_data(
            syncpulse_files=syncpulse_files, video_files=video_files, neural_files=neural_files,
            sleepstate_files=sleepstate_files,
            neural_bin_files_per_sleepstate_file=neural_bin_files_per_sleepstate_file,
            manual_video_neural_offset_sec=manual_video_neural_offset_sec,
            n_channels=n_channels, fs=fs,
            initial_neural_file_for_sleepstate_mapping=initial_neural_file_for_sleepstate_mapping,
            ignore_ecube_deviation_sec=ignore_ecube_deviation_sec,
        )
    print('NTK Sync ran successfully. Uploading results...')

    # Account for no sleep state data
    # Account for sleep state == 0, treat as invalid
    mask_sleep_state = np.ones_like(output_matrix, dtype=np.bool) if include_unlabeled \
        else output_matrix['sleep_state'] > 0
    mask_neural_filename_ix = output_matrix['neural_filename_ix'] >= 0
    valid = np.logical_and(mask_sleep_state, mask_neural_filename_ix)

    n_samples = np.sum(valid)
    assert n_samples > 0, 'No samples found.'

    labels_structured_array_dtypes = [
        ('activity', np.int8),      # -1=no data; 0=none; 1=moving; 2=grooming; 3=drinking
        ('sleep_state', np.int8),   # 1=wake/active; 2=nrem; 3=rem; 4=micro-arousal; 5=wake/passive; 100=unspecified
        ('next_wake_state', np.int64),  # time to (next|last) (wake|nrem|rem) sleep state in
        ('next_nrem_state', np.int64),  # n_samples (e.g. 1 = 1/25000ths of a sec) or -1 if current state is
        ('next_rem_state', np.int64),   # wake|nrem|rem respectively or -2 if end of sequence is reached
        ('last_wake_state', np.int64),  #
        ('last_nrem_state', np.int64),  #
        ('last_rem_state', np.int64),   #
        ('video_filename_ix', np.int32),
        ('video_frame_offset', np.int32),
        ('neural_filename_ix', np.int32),
        ('neural_offset', np.int64),
    ]

    labels_matrix = np.empty(shape=(n_samples,), dtype=labels_structured_array_dtypes)

    labels_matrix['activity'] = -1  # reserved for future use
    labels_matrix['sleep_state'] = output_matrix['sleep_state'][valid]
    labels_matrix['video_filename_ix'] = output_matrix['video_filename_ix'][valid]
    labels_matrix['video_frame_offset'] = output_matrix['video_frame_offset'][valid]
    labels_matrix['neural_filename_ix'] = output_matrix['neural_filename_ix'][valid]
    labels_matrix['neural_offset'] = output_matrix['neural_offset'][valid]

    assert np.min(labels_matrix['neural_filename_ix']) >= 0
    assert np.max(labels_matrix['neural_filename_ix']) < len(neural_files)

    # Split labels_matrix on discontinuities & compute time from last state going forward and back in time
    contiguous_segments = np.split(
        output_matrix[valid]['sleep_state'],
        np.where(output_matrix[valid]['video_frame_global_ix'][1:] -
                 output_matrix[valid]['video_frame_global_ix'][:-1] > 1)[0] + 1
    )
    wake_fwd, wake_bwd = zip(*[compute_time_from_last_label(s, [1, 5]) for s in contiguous_segments])
    nrem_fwd, nrem_bwd = zip(*[compute_time_from_last_label(s, [2]) for s in contiguous_segments])
    rem_fwd, rem_bwd = zip(*[compute_time_from_last_label(s, [3]) for s in contiguous_segments])
    labels_matrix['next_wake_state'] = np.concatenate(wake_fwd)
    labels_matrix['last_wake_state'] = np.concatenate(wake_bwd)
    labels_matrix['next_nrem_state'] = np.concatenate(nrem_fwd)
    labels_matrix['last_nrem_state'] = np.concatenate(nrem_bwd)
    labels_matrix['next_rem_state'] = np.concatenate(rem_fwd)
    labels_matrix['last_rem_state'] = np.concatenate(rem_bwd)

    video_files = [os.path.basename(vf) for vf in video_files]

    output = os.path.splitext(output)[0]
    output_csv = '{}.csv'.format(output)
    output_npz = '{}.npz'.format(output)
    tmp_csv = '/tmp/{}'.format(uuid.uuid1())
    tmp_npz = '/tmp/{}'.format(uuid.uuid1())

    with open(tmp_csv, 'w+') as f_csv, open(tmp_npz, 'wb+') as f_npz:
        np.savetxt(f_csv, labels_matrix, delimiter=',', header=','.join([x[0] for x in labels_structured_array_dtypes]),
                   fmt=['%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d'])
        np.savez(file=f_npz, labels_matrix=labels_matrix, video_files=video_files, neural_files=neural_files)
    tf.io.gfile.copy(tmp_csv, output_csv, overwrite=overwrite)
    tf.io.gfile.copy(tmp_npz, output_npz, overwrite=overwrite)
    os.remove(tmp_csv)
    os.remove(tmp_npz)
    print('Saved ', output_csv)
    print('Saved ', output_npz)
    print('Label generation complete.')


def compute_time_from_last_label(sequence: np.ndarray, labels: list):
    """
    Compute the label for predicting sleep state changes forward and backward in time.
    Time label value definition:
        -2: No state change occurs within a continuous segment from this point
        -1: Means this is the current state
        0+: The time in fps (15 fps in current cases). 1 sec = 15
    """
    results = []

    for reverse in [True, False]:
        seq = np.flip(sequence) if reverse else sequence
        result = np.empty_like(seq, dtype=np.int64)
        count = -1

        for ix, s in enumerate(seq):
            if s in labels:
                result[ix] = -1
                count = 0
            elif count == -1:
                result[ix] = -2
                # count = 0
            else:
                result[ix] = count + 1
                count += 1
        result = np.flip(result) if reverse else result
        results.append(result)

    return tuple(results)


if __name__ == '__main__':
    main(**parse_args())
