""" Write neural data files and labels for splicing positive-control experiment. """
import numpy as np
import argparse
import smart_open
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_labels_file', type=str,
        help='Original labels file location (local or S3).',
    )
    parser.add_argument(
        '--output_labels_file', type=str,
        help='Output location of updated labels file.',
    )
    parser.add_argument(
        '--input_neural_files_path', type=str,
        help='Path to original neural files.',
    )
    parser.add_argument(
        '--output_neural_files_prefix', type=str, default='splice',
        help='Relative output path for modified neural files relative to input_neural_files_path. Default "/splice/"',
    )
    parser.add_argument(
        '--name', type=str, help='Unique name to append to files to identify experiment.'
    )
    parser.add_argument(
        '--segment_target_begin', type=int, help='Start of segment to be replaced by row number in labels matrix.',
    )
    parser.add_argument(
        '--segment_source_begin', type=int, help='Start of segment to use as the source by row number in labels matrix.'
    )
    parser.add_argument(
        '--segment_length', type=int, help='Length of segment to replace by number of rows in labels matrix.'
    )
    parser.add_argument(
        '--n_channels', type=int, help='Number of channels in the dataset.'
    )
    parser.add_argument(
        '--replace_method', type=str, help='Replace method, one of: [step-function|logistic-merge].'
    )
    parser.add_argument(
        '--logistic_k', type=float, default=0.006, required=False,
        help='Logistic function k parameter, or steepness of the sigmoid curve. '
             'Only valid when logistic-merge method is chosen.'
    )
    parser.add_argument(
        '--logistic_x0', type=int, default=833, required=False,
        help='Logistic function sigmoid curve right shift. '
             'Only valid when logistic-merge method is chosen.'
    )
    parser.add_argument(
        '--context_only', type=int, default=-1, required=False,
        help='Only save labels that are being modified + this number of samples before and after the updated segment.'
             '-1 to disable context, and save all labels (default), 0 saves only changed labels.'
    )
    parser.add_argument(
        '--labels_multiplier', type=int, default=1, required=False,
        help='Increases the frequency of labels by X times. Example: 4 would create 4x as many labels. '
             'If video FPS is 15, then normally there is a label every 1/15th of a second, 4x increase would mean '
             'a label occurs every 1/60th of a second.'
    )

    return vars(parser.parse_args())


def load_labels_matrix(input_labels_file):
    # Load input labels file
    with smart_open.open(input_labels_file, 'rb') as f:
        npz = np.load(f)
        labels_matrix = npz['labels_matrix']
        video_files = npz['video_files']
        neural_files = npz['neural_files']

    return labels_matrix, video_files, neural_files


def generate_labels(labels_matrix, video_files, neural_files, output_labels_file, output_neural_files_prefix,
                    name, segment_target_begin, segment_source_begin, segment_length, context_only, labels_multiplier):

    with smart_open.open(output_labels_file, 'wb') as f:
        # Update ground truth in labels file
        fields = ['sleep_state', 'video_filename_ix', 'video_frame_offset']
        labels_matrix[fields][segment_target_begin:segment_target_begin + segment_length] = \
            labels_matrix[fields][segment_source_begin:segment_source_begin + segment_length]

        # Update neural_files with new path/filename of updated neural file "splice/neural_file_name-${name}.bin"
        neural_filename_ix = np.unique(labels_matrix['neural_filename_ix'][segment_target_begin:segment_target_begin + segment_length])
        assert neural_filename_ix.shape[0] == 1
        old_name = neural_files[neural_filename_ix[0]]
        new_name = f'{output_neural_files_prefix}/{name}-{old_name}'
        print(f'DEBUG> Generating new neural file name: {new_name}')

        str_len = max(neural_files.dtype.itemsize / np.dtype('U1').itemsize, len(new_name))
        neural_files = np.array(neural_files, dtype=f'U{str_len}')
        neural_files[neural_filename_ix] = new_name

        # Filter only labels if context-only enabled
        if context_only >= 0:
            labels_matrix = labels_matrix[max(0, segment_target_begin - context_only):segment_target_begin + segment_length + context_only]
            print(f'DEBUG> Filtered labels matrix down to size {labels_matrix.shape} for context_only value of {context_only} from segment {max(0, segment_target_begin - context_only)} to {segment_target_begin + segment_length + context_only}')

        # Multiply the number of labels by labels_multiplier - used to increase resolution of splice experiment
        if labels_multiplier > 1:
            labels_matrix_expanded = np.zeros_like(labels_matrix, shape=(labels_matrix.shape[0] * labels_multiplier))
            neural_diff = labels_matrix[1]['neural_offset'] - labels_matrix[0]['neural_offset']
            for i, label in enumerate(labels_matrix):
                neural_offsets = label['neural_offset'] - np.linspace(0, neural_diff, labels_multiplier + 1)[:-1].astype(np.int32)
                for j, neural_offset in enumerate(np.flipud(neural_offsets)):
                    label_copy = label.copy()
                    label_copy['neural_offset'] = neural_offset
                    labels_matrix_expanded[labels_multiplier * i + j] = label_copy

        # Save new labels file, updated by name parameter
        np.savez(f, labels_matrix=labels_matrix, video_files=video_files, neural_files=neural_files)
        print(f'Saved: {output_labels_file}')

    return labels_matrix, old_name


def generate_neural_data(labels_matrix, neural_files, input_neural_files_path, output_neural_files_prefix,
                         name, segment_target_begin, segment_source_begin, segment_length, n_channels,
                         replace_method, logistic_k, logistic_x0):
    """

    :param labels_matrix:
    :param neural_files:
    :param input_neural_files_path:
    :param output_neural_files_prefix:
    :param name:
    :param segment_target_begin: measured in labels_matrix row number
    :param segment_source_begin: measured in labels_matrix row number
    :param segment_length: measured in labels_matrix row numbers
    :param n_channels:
    :param replace_method:
    :param logistic_k:
    :param logistic_x0:
    :return:
    """
    fps = 15    # 15 frames per second fixed
    fs = 25000  # sampling rate fixed to 25kHz

    source_neural_filename_ix = np.unique(labels_matrix['neural_filename_ix'][segment_source_begin:segment_source_begin + segment_length])
    target_neural_filename_ix = np.unique(labels_matrix['neural_filename_ix'][segment_target_begin:segment_target_begin + segment_length])
    assert source_neural_filename_ix.shape[0] == 1
    assert target_neural_filename_ix.shape[0] == 1
    source_neural_filename = os.path.join(input_neural_files_path, neural_files[source_neural_filename_ix][0])
    target_neural_filename = os.path.join(input_neural_files_path, neural_files[target_neural_filename_ix][0])
    print(f'DEBUG> source_neural_filename {source_neural_filename}\n       target_neural_filename {target_neural_filename}')

    output_filename = os.path.join(input_neural_files_path, output_neural_files_prefix, f'{name}-{neural_files[target_neural_filename_ix][0]}')
    print(f'DEBUG> output_filename: {output_filename}')

    # compute the range of for the segment to copy in bytes and sample-count
    source_samples_from = labels_matrix[segment_source_begin]['neural_offset']
    target_samples_from = labels_matrix[segment_target_begin]['neural_offset']
    source_bytes_from = 8 + source_samples_from * 2 * n_channels
    target_bytes_from = 8 + target_samples_from * 2 * n_channels
    segment_length_samples = int((segment_length / fps) * fs)
    segment_length_bytes = segment_length_samples * 2 * n_channels

    # open the source, target, and output neural data file
    with smart_open.open(source_neural_filename, 'rb') as source_file, \
            smart_open.open(target_neural_filename, 'rb') as target_file, \
            smart_open.open(output_filename, 'wb') as output_file:

        # copy the first target_bytes_from from target_file to the output, before the injected section
        # this includes the timestamp offset
        # perform this operation in chunks of at most 1MB
        print(f'DEBUG> Copying first bytes from target file to output file {target_neural_filename}...', end='')
        for b in range(0, target_bytes_from, 2**20):
            print('.', end='')
            output_file.write(target_file.read(min(2**20, target_bytes_from - b)))
        print('')

        # read data segment
        source_file.seek(source_bytes_from)
        print(f'DEBUG> Reading source segment from {source_neural_filename} of size {segment_length_bytes} bytes')
        source_data_bytes = source_file.read(segment_length_bytes)
        print(f'DEBUG> Reading target segment from {target_neural_filename} of size {segment_length_bytes} bytes')
        target_data_bytes = target_file.read(segment_length_bytes)
        print(f'DEBUG> source_data_bytes: {len(source_data_bytes)}, target_data_bytes {len(target_data_bytes)}, segment_length_bytes {segment_length_bytes}')
        source_data_npy = np.frombuffer(source_data_bytes, dtype=np.int16).reshape((n_channels, -1), order='F')
        target_data_npy = np.frombuffer(target_data_bytes, dtype=np.int16).reshape((n_channels, -1), order='F')
        print(f'DEBUG> source_data_npy {source_data_npy.shape}, target_data_npy {target_data_npy.shape}')

        # merge source and target data
        if replace_method == 'step-function':
            data = source_data_bytes
        elif replace_method == 'logistic-merge':
            weight = mirrored_logistic_function(segment_length_samples, logistic_k, logistic_x0)
            print(f'DEBUG> weight.shape {weight.shape}, source_data_npy.dtype {source_data_npy.dtype}')
            data_npy_merged = source_data_npy * weight + target_data_npy * (1 - weight)  # multiplication relies on broadcasting
            print(f'DEBUG> data_npy_merged type {data_npy_merged.dtype}')
            data = np.rint(data_npy_merged).astype(np.int16).tobytes(order='F')
        else:
            raise ValueError(f'{replace_method} is not a valid value for replace_method.')

        # write target segment data
        print(f'DEBUG> Writing target segment of size {len(data)}')
        output_file.write(data)

        # copy the final bytes from tar in chunks of at most 1MB
        print(f'DEBUG> Copying final bytes...', end='')
        while (b:=target_file.read(2**20)) != b'':
            print('.', end='')
            output_file.write(b)
        print('')

    print(f'Saved: {output_filename}')


def mirrored_logistic_function(segment_length, k, x0):
    x = np.arange(segment_length)
    # produces a logistic function
    s_curve = 1 / (1 + np.exp(-k * (x - x0)))
    # mirror the logistic function across the half way point
    n = s_curve.shape[0]
    mirrored_s_curve = np.concatenate((s_curve[:int(0.5*n)], s_curve[:n - int(0.5*n)][::-1]))
    mirrored_s_curve = np.expand_dims(mirrored_s_curve, axis=0)
    return mirrored_s_curve


def main():
    args = parse_args()
    labels_matrix, video_files, neural_files = load_labels_matrix(args['input_labels_file'])
    generate_labels(
        labels_matrix=labels_matrix.copy(),
        video_files=video_files,
        neural_files=neural_files.copy(),
        output_labels_file=args['output_labels_file'],
        output_neural_files_prefix=args['output_neural_files_prefix'],
        name=args['name'],
        segment_target_begin=args['segment_target_begin'],
        segment_source_begin=args['segment_source_begin'],
        segment_length=args['segment_length'],
        context_only=args['context_only'],
        labels_multiplier=args['labels_multiplier'],
    )
    generate_neural_data(
        labels_matrix=labels_matrix.copy(),
        neural_files=neural_files.copy(),
        input_neural_files_path=args['input_neural_files_path'],
        output_neural_files_prefix=args['output_neural_files_prefix'],
        name=args['name'],
        segment_target_begin=args['segment_target_begin'],
        segment_source_begin=args['segment_source_begin'],
        segment_length=args['segment_length'],
        n_channels=args['n_channels'],
        replace_method=args['replace_method'],
        logistic_k=args['logistic_k'],
        logistic_x0=args['logistic_x0'],
    )


if __name__ == '__main__':
    main()
