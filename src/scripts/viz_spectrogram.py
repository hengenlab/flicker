""" Generates the spectrogram for a set of neural data files. """
import argparse
import neuraltoolkit as ntk
import scipy.signal as signal
from typing import List
from braingeneers.utils import smart_open
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import os
import io
import gc
import psutil


open = smart_open.open
GAIN = np.float64(0.19073486328125)


def arg_parse():
    parser = argparse.ArgumentParser(description='Generate a spectrogram for a given set of neural data files.')

    parser.add_argument(
        '-f', '--file', action='append', dest='files', nargs='?', const='',
        help='A single input file, use -f multiple times to list multiple input files. Local or S3.'
    )
    parser.add_argument(
        '-n', '--nchannels', type=int, required=False,
        help='Number of channels for data files. When unset channels will be parsed from filename.'
    )
    parser.add_argument(
        '--fs', type=int, required=False, default=25000,
        help='Sampling rate of raw data files. Default: 25000'
    )
    parser.add_argument(
        '-c', '--channels', required=False,
        help='CSV list of channels to include. Defaults to plotting all channels.'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file directory.'
    )

    parsed_args = vars(parser.parse_args())
    return parsed_args


def main(files: List[str], nchannels: int, output: str, fs: int, channels: str):
    print('Processing files:\n', '\t' + '\n\t'.join(files), f'\nWriting output to path {output}')
    assert isinstance(files, list) and len(files) > 0, 'No files provides using --file'
    nchannels = int(files[0].split('_')[-5]) if nchannels is None else nchannels
    fsd = 200
    channels_to_plot = range(nchannels) if channels is None or channels == '' else list(map(int, channels.split(',')))
    print(f'Plotting channels {list(channels_to_plot)}')

    # Load data
    data_downsampled = []

    for c, filename in enumerate([f for f in files if f != '']):
        with open(filename, 'rb') as f:
            # data = np.fromfile(f, dtype=np.int16, count=-1, offset=8).reshape((nchannels, -1), order='F')
            f.seek(8)
            data = np.frombuffer(f.read(), dtype=np.int16).reshape((nchannels, -1), order='F')[:, ::np.int64(fs/fsd)].copy()
            print(f'Data loaded for {filename}')
            # Downsample data to 200hz
            data_downsampled.append(data)
            gc.collect()

    data_downsampled = np.hstack(data_downsampled)

    #
    # Plots
    #
    for ci, channel in enumerate(channels_to_plot):
        #
        # Spectrogram
        #
        downdatlfp_chan = data_downsampled[channel, :]
        emg_amp = downdatlfp_chan  # emg_amp is the blue line, which was originally an EMG trace, but now I use it as the total bandpower from the downdatlfp trace

        print('Calculating bandpower...')
        f, t_spec, x_spec = signal.spectrogram(downdatlfp_chan, fs=fsd, window='hanning', nperseg=1000, noverlap=1000 - 1, mode='psd')
        fmax = 64
        fmin = 1
        x_mesh, y_mesh = np.meshgrid(t_spec, f[(f <= fmax) & (f >= fmin)])
        plt.figure(figsize=(16, 2))
        plt.pcolormesh(x_mesh, y_mesh, np.log10(x_spec[(f <= fmax) & (f >= fmin)]), cmap='jet')

        fsemg = 1
        realtime = np.arange(np.size(emg_amp)) / fsemg
        plt.plot(realtime, (emg_amp - np.nanmean(emg_amp)) / np.nanstd(emg_amp))
        plt.ylim(1, 64)
        plt.xlim(0, 3600)
        plt.yscale('log')
        plt.title(f'Channel {channel} (non chan map) - Time Period {os.path.basename(files[0])[:-4]} - {os.path.basename(files[-1])[:-4]}')

        #
        # Save
        #
        output_filename = os.path.join(output, f'spectrogram_hsv_{os.path.basename(files[0])}_chan_{channel}.png')
        with open(output_filename, 'wb') as f:
            plt.savefig(f, format='png')
        print(f'Saved {output_filename}')
        plt.close()
        gc.collect()

        print(f'Finished channel {channel}')
        print(f'Mem usage for chan {channel}: {psutil.virtual_memory()}')


if __name__ == '__main__':
    main(**arg_parse())
