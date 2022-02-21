""" A tensorflow dataset for Hengenlab mouse data with videos pre-converted to per-frame image files. """
import tensorflow.compat.v1 as tf
import os
import importlib
import heapq
import queue
import traceback
import numpy as np
import numpy.lib.recfunctions as recfunctions
from PIL import Image
import logging
import asyncs3.s3readerv5 as s3reader
import multiprocessing
import io
import configparser
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from fs.osfs import OSFS
import urllib
import itertools
from functools import lru_cache
import re
import time
import boto3
from botocore.exceptions import ClientError
import math
import concurrent.futures
from tenacity import *
import types
from braingeneers.utils import smart_open
import common_py_utils.common_utils as common_utils
from recordtype import recordtype
import atexit
import warnings
import inspect
from common_py_utils import yaml_cfg_parser
import humanize


warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", ResourceWarning)

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SHAPE = (120, 160, 3)
IMG_SHAPE_PADDED = (128, 160, 3)
IMG_SHAPE_RAW = (480, 640, 3)


# a tuple to store the index into labels_matrix and a boolean indicating if the sample targets an objective
IxTarget = recordtype(
    'IxTarget', ['ix', 'target_wnr', 'target_ma', 'target_pa', 'target_time', 'time_as_sampled']
)


class DatasetBase:
    def as_dataset(self):
        """ Returns the tensorflow dataset object. """
        raise NotImplemented('Subclasses must implement this method.')

    def shutdown(self):
        """ Optional method called when the training completes. """
        pass


class DatasetImgPreprocessing(DatasetBase):
    """
    Performs preprocessing steps such clearing parts of the video, batching, etc.

    Notes: There are significant challenges in dynamically feeding data to a tensorflow dataset.
    this class has been re-written to support both tensorflow dataset operations (meant for batch processing, not
    sample-by-sample), and python transformations. The latter should be used when performing multiple predictions.

    Issues with generators:  https://stackoverflow.com/questions/56939282
    Issues with keras predict: https://github.com/tensorflow/tensorflow/issues/30448
    """

    def __init__(self,
                 ds: tf.data.Dataset = None,
                 batch_size=None):
        """
        Expected input shape: {'frame', (480, 640, 3), dtype=tf.uint8}

        Notes: Initializable iterators don't work with eager execution, so the generator approach is
               used as a workaround.

        :param ds: input to this dataset, maybe a dataset such as DatasetHengenlabMiceFrameGenerator(...).as_dataset(...)
                   or it must be initialized manually if None.
        """
        self.input_dataset = ds
        self.output_dataset = None
        self.batch_size = batch_size

    def preprocessing_pipeline_py_function(self, data_eager_tensor):
        pyfunc = tf.py_function(
            func=lambda x: self.preprocessing_pipeline(x.numpy())['frame'],
            inp=[data_eager_tensor['frame']],
            Tout=[tf.float32],
            name='preprocessing_pipeline_py_function',
        )[0]
        pyfunc.set_shape(IMG_SHAPE_PADDED)

        return {'frame': pyfunc}

    def preprocessing_pipeline_batch(self, data_npy: np.ndarray):
        assert len(data_npy.shape) == 4, 'Invalid input shape {}'.format(data_npy.shape)

        batch_result = np.stack([self.preprocessing_pipeline(x)['frame'] for x in data_npy], axis=0)

        return {'frame': batch_result}

    def preprocessing_pipeline(self, data_npy: np.ndarray):
        """ Adds the preprocessing steps to the dataset pipeline """

        assert len(data_npy.shape) == 3, 'Invalid input shape {}'.format(data_npy.shape)

        sample = {'frame': data_npy}
        sample = self.img_resize(sample)
        sample = self.img_rescale(sample)
        sample = self.img_pad(sample)
        sample = self.img_black_text_area(sample)

        return sample

    def as_dataset(self):
        assert self.input_dataset is not None

        if self.output_dataset is None:
            ds = self.input_dataset.map(
                map_func=self.preprocessing_pipeline_py_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            ds = ds.batch(batch_size=self.batch_size) if self.batch_size else ds
            self.output_dataset = ds

        return self.output_dataset

    @staticmethod
    def img_black_text_area(sample):
        """ Blacks out the time text area in Hengenlab data. """
        sample['frame'][5:12, 48:112, :] = 0.0
        return sample

    @staticmethod
    def img_rescale(sample):
        """ Rescales a uint8 image to [-1, 1] float32 range. """
        resized_img = (sample['frame'].astype(np.float32) - 127.5) / 127.5
        return {'frame': resized_img}

    @staticmethod
    def img_pad(sample):
        """ Pads the additional rows of zeros between self.img_shape and self.img_shape_padded """
        pad_rows = IMG_SHAPE_PADDED[0] - IMG_SHAPE[0]
        _, w, c = sample['frame'].sz
        padded_zeros = np.zeros(shape=(pad_rows, w, c), dtype=sample['frame'].dtype)
        img_padded = np.concatenate((sample['frame'], padded_zeros), axis=0)
        return {'frame': img_padded}

    @staticmethod
    def img_resize(sample):
        """
        Resize to self.img_shape, resizing by aspect ratio.

        :param sample: image input tensor
        :return: numpy image dtype=np.uint8
        """

        # def apply(img_eager):
        img_numpy = sample['frame']
        h, w, _ = IMG_SHAPE
        pil_img = Image.fromarray(img_numpy)
        pil_img = pil_img.resize(size=(w, h), resample=Image.NEAREST)
        return {'frame': np.array(pil_img)}


class DatasetEndecodeFrames(DatasetBase):
    """
    Endecode model frame reader dataset.

    To be done: Change filter_video_filename_ix and filter_exclude_include to the format used by DatasetSleepStateLabels
                which uses the more simple include_neural_files exclude_neural_files
    """

    def __init__(self, labels_endecode_file, max_async_ops=100, maintain_order=False, interleave=True,
                 filter_video_filename_ix: (tuple, list) = (), filter_exclude_include: str = 'exclude',
                 resolution_width_height: (int, int) = IMG_SHAPE[:2],
                 s3_basepath: str = 's3://braingeneers/dfparks/EAB40/Video_Frames/',
                 endpoint_url: str = 'https://s3.nautilus.optiputer.net',
                 debug=False):
        assert filter_exclude_include in ['exclude', 'include'], 'Valid values are "exclude" or "include".'
        self.resolution_width_height = resolution_width_height
        self.interleave = interleave
        self.max_async_ops = max_async_ops
        self.maintain_order = maintain_order
        self.s3_basepath = s3_basepath
        self.endpoint_url = endpoint_url
        self.debug = debug

        with smart_open.open(labels_endecode_file, 'rb') as f, np.load(f) as npz:
            self.labels_endecode = npz['labels_neuralendecode']
            self.video_files = npz['video_files']

        # Create a filter mask to include or exclude labels based on video_filename_ix
        self.valid_labels_mask = np.ones(shape=self.labels_endecode.sz, dtype=np.bool) \
            if filter_exclude_include == 'exclude' \
            else np.zeros(shape=self.labels_endecode.sz, dtype=np.bool)
        for ix in filter_video_filename_ix:
            if filter_exclude_include == 'exclude':
                self.valid_labels_mask = ~(self.labels_endecode['video_filename_ix'] == ix) & self.valid_labels_mask
                logging.info('Excluding video_filename_ix %s', ix)
            if filter_exclude_include == 'include':
                self.valid_labels_mask = (self.labels_endecode['video_filename_ix'] == ix) | self.valid_labels_mask
                logging.info('Including video_filename_ix %s', ix)
        self.labels_endecode = self.labels_endecode[self.valid_labels_mask]

    def as_dataset(self):
        # auto = tf.data.experimental.AUTOTUNE

        ds = tf.data.Dataset.from_generator(
            self.generate,
            output_types={'frame': tf.uint8},
            output_shapes={'frame': IMG_SHAPE},
        )

        return ds

    def generate(self):
        # Produce cluster bincounts from labels
        cluster_ix_interleaved = interleave_samples(self.labels_endecode['cluster_ix']) if self.interleave \
            else np.arange(self.labels_endecode['cluster_ix'].sz[0])

        with s3reader.S3Reader(endpoint_url=self.endpoint_url,
                               maintain_order=self.maintain_order,
                               debug=self.debug) as reader:
            # seed the IO queue
            for sample in self.labels_endecode[cluster_ix_interleaved][:self.max_async_ops]:
                reader.submit(self.s3_read, reader, sample, self.video_files[sample['video_filename_ix']],
                              self.s3_basepath)

            # process results and feed queue
            for sample in self.labels_endecode[cluster_ix_interleaved][self.max_async_ops:]:
                try:
                    b = reader.get()
                except GeneratorExit as ge:
                    raise ge
                except Exception as e:
                    print('Exception encountered in TF dataset: {} for sample {}'.format(e, sample))
                    continue

                reader.submit(self.s3_read, reader, sample, self.video_files[sample['video_filename_ix']],
                              self.s3_basepath)
                yield {'frame': b}

            # drain the remaining results
            for _ in range(self.max_async_ops):
                try:
                    b = reader.get()
                except GeneratorExit as ge:
                    raise ge
                except Exception as e:
                    print('Exception encountered in TF dataset: {} for sample {}'.format(e, sample))
                    continue

                yield {'frame': b}

    @staticmethod
    async def s3_read(reader, sample, video_filename, s3_basepath):
        # url = 's3://braingeneers/dfparks/EAB40/Video_Frames_Bin/frames_' + os.path.basename(video_filename) + '.bin'
        url = os.path.join(s3_basepath, 'frames_{}.bin'.format(os.path.basename(video_filename)))
        data_offset = sample['frame_data_offset']
        data_length = sample['frame_data_length']
        f = reader.open.open(url, awaitable=True)
        f.seek(data_offset)
        b = await f.read(data_length)

        b_io = io.BytesIO(b)
        img_pil = Image.open(b_io)
        img_pil.verify()
        b_io.seek(0)
        img_pil = Image.open(b_io)
        img_npy = np.array(img_pil)

        return img_npy


class DatasetNeuralSequenceLabels(DatasetBase):
    def __init__(self, labels_neuralendecode: (str, list, tuple), neural_files_basepath: str,
                 prefix_sample_length: int, sequence_length: int,
                 include_neural_files: (tuple, list) = None, exclude_neural_files: (tuple, list) = None,
                 include_video_files: (tuple, list) = None, exclude_video_files: (tuple, list) = None,
                 n_channels: int = None, fs: int = 25000):
        """
        Output keys:
            {'DATASET_NAME', 'latent_state_row_index',
            'video_filename_ix', 'video_filename', 'video_frame_offset', 'frame_data_offset', 'frame_data_length',
            'neural_filename_ix', 'neural_filename', 'neural_offset',
            'cluster_ix'}
        :param labels_neuralendecode: path (local or S3) to labels_neuralendecode.npz file
        :param neural_files_basepath: base path to the where neural filenames in the labels file can be found,
               the neural file sizes will be read from here to validate sequence lengths.
        :param prefix_sample_length: number of neural samples to include before the first video frame, in this
               dataset that only limits samples at the beginning of neural data.
        :param sequence_length: The number of delta_latent_state values in the sequence.
        :param include_neural_files: A (tuple or list) of (ints or strings), when provided only these neural
               files will be sampled from. Mutually exclusive with exclude_neural_files.
        :param exclude_neural_files: A (tuple or list) of (ints or strings), when provided all neural files
               except these will be sampled from. Mutually exclusive with include_neural_files.
        :param include_video_files: A (tuple or list) of (ints or strings), when provided only these video
               files will be sampled from. Mutually exclusive with exclude_video_files.
        :param exclude_video_files: A (tuple or list) of (ints or strings), when provided all video files
               except these will be sampled from. Mutually exclusive with include_video_files.
        :param n_channels: None == autodetect. Number of channels in data files, if unspecified (None) the
               number of channels will be inferred from the filename of the first neural .bin file.
        :param fs: sampling rate, 25000 default
        """
        labels_neuralendecode = \
            [labels_neuralendecode] if isinstance(labels_neuralendecode, str) else labels_neuralendecode

        # Merge all labels_neuralendecode files and drop irrelevant data
        dtype = [
            ('dataset_names_ix', '<u4'),
            ('latent_state_row_index', '<u4'),
            ('video_filename_ix', '<i4'),
            ('video_frame_offset', '<u4'),
            ('frame_data_offset', '<u8'),
            ('frame_data_length', '<u4'),
            ('neural_filename_ix', '<i4'),
            ('neural_offset', '<i8'),
            ('cluster_ix', '<i4'),
        ]
        self.labels_neuralendecode_merged = []
        self.dataset_names = []
        self.video_files = []
        self.neural_files = []
        self.neural_files_sample_lengths = []
        self.n_channels = parse_n_channels(n_channels, self.neural_files[0]) if n_channels is None else n_channels
        self.prefix_sample_length = prefix_sample_length
        self.sequence_length = sequence_length
        self.fs = fs

        for label_file in labels_neuralendecode:
            with tf.gfile.GFile(label_file, mode='rb') as f, np.load(f) as npz:
                labels = npz['labels_neuralendecode']
                video_files = npz['video_files']
                neural_files = npz['neural_files']

                data = np.empty(shape=(labels.sz[0],), dtype=dtype)
                data['dataset_names_ix'] = len(self.labels_neuralendecode_merged)
                data['latent_state_row_index'] = np.arange(0, labels.sz[0])
                data['video_filename_ix'] = labels['video_filename_ix'] + len(self.video_files)
                data['neural_filename_ix'] = labels['neural_filename_ix'] + len(self.neural_files)
                for field in ['video_frame_offset', 'frame_data_offset', 'frame_data_offset', 'frame_data_length',
                              'neural_offset', 'cluster_ix']:
                    data[field] = labels[field]

                self.labels_neuralendecode_merged.append(data)
                self.dataset_names.append(re.search(r'.*_([^.]*).*?', label_file).group(1))
                self.video_files.extend(video_files)
                self.neural_files.extend(neural_files)
                self.neural_files_sample_lengths.extend(get_neural_file_sample_counts(
                    neural_files, neural_files_basepath, self.n_channels
                ))

        self.labels_neuralendecode_merged = np.concatenate(self.labels_neuralendecode_merged)

        # Process include/exclude filters and exclude samples that would have insufficient prefix_sample_length neural data
        # Validate mutual exclusivity and dtypes
        assert_mutually_exclusive(valid_dtypes=(int, str), include_neural_files=include_neural_files,
                                  exclude_neural_files=exclude_neural_files)
        assert_mutually_exclusive(valid_dtypes=(int, str), include_video_files=include_video_files,
                                  exclude_video_files=exclude_video_files)

        # Validate input names for [include|exclude]_[neural|video]_files exist in the label files
        mask_neural_files = include_exclude_mask(self.neural_files,
                                                 self.labels_neuralendecode_merged['neural_filename_ix'],
                                                 include_neural_files, exclude_neural_files)
        mask_video_files = include_exclude_mask(self.video_files,
                                                self.labels_neuralendecode_merged['video_filename_ix'],
                                                include_video_files, exclude_video_files)

        # Exclude samples that don't have sufficient neural data going back prefix_sample_length
        unique_neural_filename_ixs = np.unique(self.labels_neuralendecode_merged['neural_filename_ix'])
        mask_no_crossing_boundary = ~(
                    self.labels_neuralendecode_merged['neural_offset'] - self.prefix_sample_length < 0)
        mask_has_prior_neural_file = np.isin(self.labels_neuralendecode_merged['neural_filename_ix'] - 1,
                                             unique_neural_filename_ixs)
        mask_prefix_sample_length = mask_no_crossing_boundary | mask_has_prior_neural_file

        # Exclude samples that don't have sufficient neural data going forward sequence_length * fs
        self.neural_files_sample_lengths = np.array(self.neural_files_sample_lengths)
        mask_no_crossing_boundary = self.labels_neuralendecode_merged['neural_offset'] + (
                    self.sequence_length * self.fs) < \
                                    self.neural_files_sample_lengths[
                                        self.labels_neuralendecode_merged['neural_filename_ix']]
        mask_has_next_neural_file = np.isin(self.labels_neuralendecode_merged['neural_filename_ix'] + 1,
                                            unique_neural_filename_ixs)
        mask_forward_sample_length = mask_no_crossing_boundary | mask_has_next_neural_file

        # Filter labels_matrix by the union of neural, video, and neural_data masks.
        mask = mask_neural_files & mask_video_files & mask_prefix_sample_length & mask_forward_sample_length
        self.labels_neuralendecode_merged = self.labels_neuralendecode_merged[mask]

    def as_dataset(self):
        ds = tf.data.Dataset.from_generator(
            self.generate,
            output_types={'DATASET_NAME': tf.str, 'latent_state_row_index': tf.uint32,
                          'video_filename_ix': tf.uint32, 'video_filename': tf.str, 'video_frame_offset': tf.uint32,
                          'frame_data_offset': tf.uint64, 'frame_data_length': tf.uint32,
                          'neural_filename_ix': tf.uint32, 'neural_filename': tf.str, 'neural_offset': tf.uint64,
                          'cluster_ix': tf.uint32},
            output_shapes={'DATASET_NAME': (), 'latent_state_row_index': (),
                           'video_filename_ix': (), 'video_filename': (), 'video_frame_offset': (),
                           'frame_data_offset': (), 'frame_data_length': (),
                           'neural_filename_ix': (), 'neural_filename': (), 'neural_offset': (),
                           'cluster_ix': ()},
        )

        return ds

    def generate(self):
        perm = np.random.permutation(self.labels_neuralendecode_merged.sz[0])

        for ix in perm:
            sample = self.labels_neuralendecode_merged[ix].copy()

            neural_filename_ix = sample['neural_filename_ix']
            neural_offset = sample['neural_offset']
            neural_filenames = self.neural_files[sample['neural_filename_ix']]
            neural_length = self.prefix_sample_length + (self.sequence_length * self.fs)

            # if file boundary is crossed forward update the neural filenames
            if sample['neural_offset'] + (self.sequence_length * self.fs) >= self.neural_files_sample_lengths[sample['neural_filename_ix']]:
                neural_filenames = '{}:{}'.format(neural_filenames, self.neural_files[sample['neural_filename_ix'] + 1])

            # if file boundary is crossed backward we need to update the neural filename ix and offset
            if sample['neural_offset'] - self.prefix_sample_length < 0:
                neural_filename_ix -= 1
                neural_offset = self.neural_files_sample_lengths[neural_filename_ix] + \
                                (sample['neural_offset'] - self.prefix_sample_length)
                neural_filenames = '{}:{}'.format(self.neural_files[neural_filename_ix], neural_filenames)

            result = {'DATASET_NAME': self.dataset_names[sample['dataset_names_ix']],
                      'video_filename': self.video_files[sample['video_filename_ix']],
                      'neural_filenames': neural_filenames,
                      'neural_filename_ix': neural_filename_ix,
                      'neural_offset': neural_offset,
                      'neural_length': neural_length,
                      **{field: sample[field] for field in [
                          'latent_state_row_index', 'video_filename_ix', 'video_frame_offset', 'frame_data_offset',
                          'frame_data_offset', 'frame_data_length', 'cluster_ix'
                      ]}}
            yield result


class DatasetDeltaLatentStateLoader(DatasetBase):
    """
    Maps a dataset with 'DATASET_NAME', and 'latent_state_row_index' to the same dataset with
    'initial_latent_state' and 'delta_latent_state' populated.
    """

    def __init__(self, latent_state: (str, list, tuple), sequence_length: int, num_parallel_calls: int = None):
        """
        :param latent_state: One or more latent state files named 'latent_state_DATASET_NAME.npy' where DATASET_NAME
        matches the dataset samples passed into this dataset.
        :param sequence_length: The number of delta_latent_state values in the sequence.
        :param num_parallel_calls: Number of parallel map calls, None=tf.data.AUTOTUNE
        """
        latent_state = [latent_state] if isinstance(latent_state, str) else latent_state
        assert isinstance(latent_state, (list, tuple))

        self.sequence_length = sequence_length
        self.num_parallel_calls = tf.data.experimental.AUTOTUNE if num_parallel_calls is None else num_parallel_calls
        self.latent_state_memmaps = {}
        self.tfs = None

        # copy s3 files to local temp filesystem if need be and populate memmaps dictionary
        for ls in latent_state:
            filename = ls
            if filename.startswith('s3://'):
                self.tfs = TempFS() if self.tfs is None else self.tfs
                with smart_open.open(filename, mode='rb') as f:
                    filename = os.path.join(self.tfs.getsyspath('/'), ls)
                    f.copy(filename)
            dataset_name = re.search(r'.*_([^.]*).*?', ls).group(1)
            self.latent_state_memmaps[dataset_name] = (filename, queue.Queue())

    def as_dataset(self, ds: tf.data.Dataset):
        """ Expecting input from DatasetNeuralSequenceLabels """
        return ds.map(self.map, num_parallel_calls=self.num_parallel_calls)

    def map(self, tensors):
        initial_latent_state, delta_latent_state = tf.py_func(
            func=self.map_delta_latent_state,
            inp=[tensors['DATASET_NAME'], tensors['latent_state_row_index']],
            Tout=(tf.float32, tf.float32),
            stateful=True,
        )

        aggregate_result = {
            **tensors,
            'initial_latent_state': initial_latent_state,
            'delta_latent_state': delta_latent_state,
        }

        return aggregate_result

    def map_delta_latent_state(self, dataset_name, latent_state_row_index):
        dataset_name = dataset_name.decode('utf-8')

        try:
            # Grab an open memmap object, multiple objects exist for parallel access, open memmap objects in the queue
            latent_state_filename, q = self.latent_state_memmaps[dataset_name]
            npy_memmap = q.get(block=False)
        except queue.Empty:
            # If no memmap objects are available for re-use open a new file pointer
            latent_state_filename, q = self.latent_state_memmaps[dataset_name]
            npy_memmap = np.load(latent_state_filename, mmap_mode='r')

        # Get the latent state from the memmap file
        initial_latent_state = npy_memmap[latent_state_row_index, :]
        delta_latent_state = npy_memmap[latent_state_row_index + 1:latent_state_row_index + self.sequence_length + 1, :] - \
                             npy_memmap[latent_state_row_index:latent_state_row_index + self.sequence_length, :]

        # Return the memmap object to the queue for reuse (avoiding the header read each time)
        self.latent_state_memmaps[dataset_name][1].put((latent_state_filename, npy_memmap))

        return initial_latent_state, delta_latent_state


class DatasetNeuralLoader(DatasetBase):
    """
    Maps a dataset with 'neural_offset', 'neural_length', and 'neural_filenames' to the same dataset
    with the 'neural_data' populated
    """

    def __init__(self, neural_files_basepath: str, n_channels: int, num_parallel_calls: int = None):

        assert isinstance(n_channels, int)

        self.neural_files_basepath = neural_files_basepath
        self.n_channels = n_channels
        # self.sample_width = sample_width
        self.num_parallel_calls = tf.data.experimental.AUTOTUNE if num_parallel_calls is None else num_parallel_calls

    def as_dataset(self, ds: tf.data.Dataset):
        num_parallel_calls = tf.data.experimental.AUTOTUNE if self.num_parallel_calls is None else self.num_parallel_calls
        ds = ds.map(self.map_py_func, num_parallel_calls=num_parallel_calls)
        return ds

    def map_py_func(self, tensors):
        neural_data = tf.py_func(
            func=self.read_neural_data,
            inp=[tensors['neural_offset'], tensors['neural_length'], tensors['neural_filenames']],
            Tout=tf.int16,
            stateful=False,
        )
        neural_data.set_shape(shape=(self.n_channels, None))
        return {'neural_data': neural_data, **tensors}

    def read_neural_data(self, neural_offset, neural_length, neural_filenames):
        neural_filenames = neural_filenames.decode('utf-8').split(':')
        neural_data = b''
        seek_offset = int(8 + neural_offset * self.n_channels * 2)
        seek_zero = 8
        read_bytes = int(neural_length * self.n_channels * 2)

        while len(neural_data) != read_bytes:
            try:
                with smart_open.open(os.path.join(self.neural_files_basepath, neural_filenames[0]), mode='rb') as f:
                    f.seek(seek_offset)
                    neural_data = f.read(read_bytes)

                if len(neural_data) < read_bytes:
                    with smart_open.open(os.path.join(self.neural_files_basepath, neural_filenames[1]),
                                           mode='rb') as f:
                        f.seek(seek_zero)
                        neural_data += f.read(read_bytes - len(neural_data))

            except Exception as e:
                print('Exception encountered continuing - neural_offset: {}, neural_length: {}, '
                      'neural_filenames: {} Exception: {}'
                      .format(neural_offset, neural_length, neural_filenames, e))
                neural_data = b''
                continue

        neural_data_int16 = np.frombuffer(neural_data, dtype=np.int16).reshape((self.n_channels, neural_length),
                                                                               order='F')
        return neural_data_int16


class DatasetBaseSharedMemory(DatasetBase):
    """ The base class for all datasets using the System V shared memory approach to distributed data loading. """

    def __init__(self, n_workers: int, subproc_properties: object):
        """
        When subclassing make sure you can super().__init__() after creating any variables
        that draw_sample will reference.

        :param n_workers: number of worker processes
        :param subproc_properties: a picklable object storing any information the subprocesses will need to generate data
        """
        self.n_workers = n_workers
        self.subproc_properties = subproc_properties

    @staticmethod
    def draw_sample(counter, subproc_properties):
        """
        Primary method to subclass.

        Returns a single sample as a dictionary. This method is run in each subclass from a subprocess
        and will be passed an incrementing counter each time it is called.

        Design patterns:
         - Random sampling: Use counter % dataset_length to determine what epoch of the dataset you're on, use
           that epoch number as a random seed to shuffle the dataset deterministically,
           return the sample at index counter % dataset_length.
        - Sequential sampling: Use counter to determine which sample to draw in sequence.

        :param: counter a unique id >= 0 indicating which sample to produce.
        :param: subproc_properties is the deserialized version of subproc_properties passed to the init method,
            this is the way to pass information to draw_sample from the main process (a labels matrix for example).
        :return: a single sample in dictionary form, None at EOF
        """
        raise NotImplemented()

    def output_types(self):
        """
        :return: a dictionary of output types matching the dictionary returned by draw_sample(...)
        """
        raise NotImplemented()

    def output_shapes(self, unspecified_dim='np'):
        """
        :return: a dictionary of output shapes matching the dictionary returned by draw_sample(...)
        """
        raise NotImplemented()

    def as_dataset(self):
        """
        Optionally subclassed.

        Creates the tensorflow dataset object. All batching, prefetching, etc should be handled here.
        Subclasses will likely want to extend or override this class.
        The default implementation assumes class level variables:
            self.batch_size
            self.output_types
            self.output_shapes
        :return: tf.data.Dataset
        """
        ds = tf.data.Dataset.from_generator(
            generator=self.sample_generator, output_types=self.output_types, output_shapes=self.output_shapes('tf')
        )
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=2)
        if self.prefetch_to_gpu:
            ds = ds.apply(tf.data.experimental.prefetch_to_device(
                device='gpu:0' if tf.test.is_gpu_available(cuda_only=True) else 'cpu', buffer_size=1
            ))
        return ds

    def sample_generator(self):
        """
        Not normally overridden by a subclass, but may be used when overriding as_dataset().

        Produces samples in the MAIN process efficiently drawing them from subprocesses via SystemV shared memory.

        If the generator is de-referenced/deleted/goes out of scope, the process pool context manager will
        close the process pool when the last worker is complete.
        """
        pool = multiprocessing.Pool(
            processes=self.n_workers,
            initializer=self._subproc_initializer,
            initargs=(self.subproc_properties,)
        )

        # shutdown function to register atexit, necessary because the finally block later is not guaranteed to be called
        def atexit_shutdown(p):
            print('atexit dataset shutting down multiprocessing pool...')
            p.close()  # this may not be called at any point when used under tensorflow
            p.terminate()
            p.join()
            print('atexit dataset multiprocessing pool shutdown successful.')
        atexit.register(atexit_shutdown, pool)

        try:
            futures_queue = []
            counter = 0

            # seed the workers
            for i in range(self.n_workers):
                futures_queue.append(pool.apply_async(
                    func=self._subproc_draw_sample,
                    args=(self.draw_sample, counter, i * 0.1))
                )
                counter += 1

            # yield samples until self.draw_sample() is None, creating a task as each one is taken
            while True:
                future = futures_queue.pop(0)
                if not future.ready():
                    t0 = time.time()
                    future.wait()
                    print(f'Dataset waited {time.time() - t0:.3f} seconds for the next sample to be ready. Increase n_workers.')
                shm_name = future.get()
                futures_queue.append(pool.apply_async(
                    func=self._subproc_draw_sample,
                    args=(self.draw_sample, counter))
                )
                counter += 1

                sample = common_utils.deserialize_shared_memory_dict(shm_name)

                if sample is not None:
                    yield sample
                else:
                    return
        finally:
            print('Dataset shutting down multiprocessing pool...')
            pool.close()  # this may not be called at any point when used under tensorflow
            pool.terminate()
            pool.join()
            print('Dataset multiprocessing pool shutdown successful.')

    @staticmethod
    def _subproc_draw_sample(draw_sample_function, counter, delayed_start=0.0):
        """ Calls self.draw_sample in a subprocess and serializes the sample using SharedMemory. """
        if delayed_start > 0.0:
            time.sleep(delayed_start)

        sample = draw_sample_function(counter, DatasetBaseSharedMemory.subproc_properties)
        shm_name = common_utils.create_shared_memory_dict(sample)
        return shm_name

    @staticmethod
    def _subproc_initializer(subproc_properties):
        """ Saves subproc_properties, an picklable object provided by subclasses """
        DatasetBaseSharedMemory.subproc_properties = subproc_properties


class DatasetMulti(DatasetBase):
    """ Wraps multiple datasets sampling randomly from each. """
    def __init__(self, parameter_files_basepath, dataset_class, multi_datasets: list, common_dataset_params: dict = {}):
        # Create each dataset object
        self.multi_datasets = []
        for md in multi_datasets:
            parameter_file = md['parameter_file']
            parameter_file = parameter_file if parameter_file.endswith('.yaml') else f'{parameter_file}.yaml'
            datasetparams = yaml_cfg_parser.parse_yaml_cfg(os.path.join(parameter_files_basepath, parameter_file), is_file=True)
            datasetparams = {**datasetparams['datasetparams'], **common_dataset_params, **md.get('dataset_params', {})}
            dataset_module = importlib.import_module('.'.join(dataset_class.split('.')[:-1]))
            ds = getattr(dataset_module, dataset_class.split('.')[-1])(**datasetparams)
            self.multi_datasets.append(ds)

    def as_dataset(self):
        return tf.data.experimental.sample_from_datasets([d.as_dataset() for d in self.multi_datasets])


# noinspection PyUnusedLocal
class DatasetSleepStateBase(DatasetBaseSharedMemory):
    """
    Shared base class for DatasetSleepStateTrain (random sampling)
    and DatasetSleepStateEvaluate (sequential sampling)
    """
    def __init__(self, labels_file: (str, tuple), sample_width_before: int, sample_width_after: int, n_channels: int,
                 batch_size: int, prefetch: int, prefetch_to_gpu: bool,
                 hp_max_predict_time_fps: int, include_modules: str,
                 neural_files_basepath: str, include_channels: (tuple, list, str) = None,
                 include_neural_files: (tuple, list) = None, exclude_neural_files: (tuple, list) = None,
                 filter_samples_near_state_change_sec: float = None,
                 n_workers: int = 1, subproc_properties: types.SimpleNamespace = None,
                 test_video_files: (tuple, list) = None, exclude_test_video_files: bool = True,
                 exclude_train_video_files: bool = False,
                 video_fps: int = 15, use_single_channel_data: int = -1, permute_samples: bool = False, **_):
        assert isinstance(exclude_test_video_files, bool)
        assert isinstance(exclude_train_video_files, bool)

        self.neural_files_basepath = neural_files_basepath
        self.sample_width_before = sample_width_before
        self.sample_width_after = sample_width_after
        self.n_channels = n_channels if use_single_channel_data == -1 else 1
        self.include_channels = common_utils.parse_include_channels(include_channels)
        self.video_fps = video_fps

        #
        # Load labels matrix
        #
        self.labels_matrix, self.video_files, self.neural_files = load_labels_file(labels_file) \
            if isinstance(labels_file, str) else labels_file

        exclude_video_files_list = []
        if exclude_test_video_files:
            exclude_video_files_list.extend(test_video_files)
        if exclude_train_video_files:
            exclude_video_files_list.extend([v for v in self.video_files.tolist() if v not in test_video_files])
        exclude_video_files_list = None if len(exclude_video_files_list) == 0 else exclude_video_files_list

        # Validate mutual exclusivity and dtypes
        assert_mutually_exclusive(valid_dtypes=(int, str), include_neural_files=None,
                                  exclude_neural_files=exclude_neural_files)
        assert_mutually_exclusive(valid_dtypes=(int, str), include_video_files=None,
                                  exclude_video_files=exclude_video_files_list)

        # if use_single_channel_data update the neural_files to use the single channel data files
        if use_single_channel_data != -1:
            self.neural_files = [
                '.'.join(nf.split('.')[:-1] + [f'channel-{use_single_channel_data}.bin'])
                for nf in self.neural_files
            ]
            exclude_neural_files = [
                '.'.join(nf.split('.')[:-1] + [f'channel-{use_single_channel_data}.bin'])
                for nf in exclude_neural_files
            ] if exclude_neural_files is not None else exclude_neural_files

        #
        # Filter labels
        #
        self.neural_sample_counts = get_neural_file_sample_counts(
            neural_files=self.neural_files, neural_files_basepath=neural_files_basepath, n_channels=self.n_channels
        )

        self.labels_matrix = self.filter_labels_matrix(
            self.labels_matrix, self.video_files, self.neural_files,
            self.neural_sample_counts, self.sample_width_before, self.sample_width_after,
            include_neural_files, exclude_neural_files, None, exclude_video_files_list,
        )

        #
        # compute label_time column, label_time is a column in the [-1, 1] range where -1 is current sleep with
        # >= X sec until a wake state, 0 is the transition sleep/wake, +1 is >= X sec until next sleep change.
        #
        self.labels_computed = np.zeros(shape=len(self.labels_matrix), dtype=[
            ('target_wnr', np.bool),                # set at runtime, not a pre-computed value
            ('target_ma', np.bool),                 # set at runtime, not a pre-computed value
            ('target_pa', np.bool),                 # set at runtime, not a pre-computed value
            ('target_time', np.bool),               # set at runtime, not a pre-computed value
            ('label_time', np.float32),             # original time label
            ('label_time_as_sampled', np.float32),  # set at runtime, not a pre-computed value
            ('label_time_sort_ixs', np.int64),      # argsort ix of label_time for efficient lookup
        ])
        self.labels_computed['label_time'] -= 2  # begin with all samples invalidated

        # replace -2 & -1 values with ~inf so minimum works
        labels_matrix_winf = self.labels_matrix.copy()
        for time_field in ['next_wake_state', 'last_wake_state', 'next_nrem_state', 'last_nrem_state', 'next_rem_state', 'last_rem_state']:
            invalid_mask = labels_matrix_winf[time_field] < 0
            labels_matrix_winf[time_field][invalid_mask] = np.iinfo(labels_matrix_winf[time_field].dtype).max

        self.labels_computed['label_time'] = np.amin([
            labels_matrix_winf['next_wake_state'],
            labels_matrix_winf['last_wake_state'],
            labels_matrix_winf['next_nrem_state'],
            labels_matrix_winf['last_nrem_state'],
            labels_matrix_winf['next_rem_state'],
            labels_matrix_winf['last_rem_state'],
        ], axis=0) / video_fps

        self.labels_computed['label_time_sort_ixs'] = np.argsort(self.labels_computed['label_time'])

        # Filter out samples near state transitions if the option is selected
        if filter_samples_near_state_change_sec is not None:
            time_threshold = filter_samples_near_state_change_sec
            mask_by_time = np.abs(self.labels_computed['label_time']) > time_threshold
            self.labels_matrix = self.labels_matrix[mask_by_time]
            self.labels_computed = self.labels_computed[mask_by_time]

        # read only matrix after creation to avoid subtle bugs, make a copy of the data if needed
        self.labels_computed.setflags(write=False)
        self.labels_matrix.setflags(write=False)

        self.batch_size = batch_size
        self.prefetch = prefetch
        self.prefetch_to_gpu = prefetch_to_gpu

        self.include_modules = include_modules
        self.ixs = compute_label_ixs(self.labels_matrix)

        self.n_workers = n_workers

        subproc_properties = types.SimpleNamespace(
            **vars(subproc_properties or types.SimpleNamespace()),
            labels_matrix=self.labels_matrix,
            labels_computed=self.labels_computed,
            ixs=self.ixs,
            batch_size=self.batch_size,
            include_modules=self.include_modules,
            neural_files_basepath=self.neural_files_basepath,
            sample_width_before=self.sample_width_before,
            sample_width_after=self.sample_width_after,
            n_channels=self.n_channels if use_single_channel_data == -1 else 1,
            include_channels=self.include_channels if use_single_channel_data == -1 else [0],
            video_files=self.video_files,
            neural_files=self.neural_files,
            neural_sample_counts=self.neural_sample_counts,
            output_shapes=self.output_shapes('np'),
            s3client=None,  # lazy init
            permute_samples=permute_samples,
        )
        super().__init__(self.n_workers, subproc_properties)

    @staticmethod
    def filter_labels_matrix(labels_matrix, video_files, neural_files,
                             neural_sample_counts, sample_width_before, sample_width_after,
                             include_neural_files, exclude_neural_files,
                             include_video_files, exclude_video_files):
        # Validate input names for [include|exclude]_[neural|video]_files exist in the label files
        mask_neural_files = include_exclude_mask(neural_files, labels_matrix['neural_filename_ix'],
                                                 include_neural_files, exclude_neural_files)
        mask_video_files = include_exclude_mask(video_files, labels_matrix['video_filename_ix'],
                                                include_video_files, exclude_video_files)

        unique_neural_filename_ixs = np.unique(
            labels_matrix['neural_filename_ix'][mask_neural_files & mask_video_files])

        # Exclude samples that don't have sufficient neural data going backward sample_width_before
        mask_no_crossing_boundary = labels_matrix['neural_offset'].astype(np.int64) - sample_width_before >= 0
        mask_has_prev_neural_file = np.isin(labels_matrix['neural_filename_ix'] - 1, unique_neural_filename_ixs)
        mask_valid_sample_length_before = mask_no_crossing_boundary | mask_has_prev_neural_file

        # Exclude samples that don't have sufficient neural data going forward sample_width_after
        mask_no_crossing_boundary = labels_matrix['neural_offset'].astype(np.int64) + sample_width_after < neural_sample_counts['neural_samples'][labels_matrix['neural_filename_ix']]
        mask_has_next_neural_file = np.isin(labels_matrix['neural_filename_ix'] + 1, unique_neural_filename_ixs)
        mask_valid_sample_length_after = mask_no_crossing_boundary | mask_has_next_neural_file

        # Filter labels_matrix by the union of neural and video mask.
        mask = mask_neural_files & mask_video_files & mask_valid_sample_length_before & mask_valid_sample_length_after
        labels_matrix_filtered = labels_matrix[mask]
        assert len(labels_matrix_filtered) > 0, 'No samples found after filtering.'
        return labels_matrix_filtered

    @property
    def output_types(self):
        output_t = {
            'sleep_states': np.int8,
            'label_time': np.float32,
            'label_time_as_sampled': np.float32,
            'next_wake_state': np.int64,
            'next_nrem_state': np.int64,
            'next_rem_state': np.int64,
            'last_wake_state': np.int64,
            'last_nrem_state': np.int64,
            'last_rem_state': np.int64,
            'target_wnr': np.bool,
            'target_ma': np.bool,
            'target_pa': np.bool,
            'target_time': np.bool,
            'neural_length': np.int32,
            'neural_data_offsets': np.int32,
            'neural_data': np.int16,
            'video_frame_offsets': np.int64,
            'neural_offsets': np.int64,
            'video_filename_ixs': np.int32,
            'video_filenames': tf.string,
            'neural_filename_ixs': np.int32,
            'neural_filenames': tf.string,
            'sample_ix': np.int32
        }
        return output_t

    @staticmethod
    @retry(wait=wait_exponential(multiplier=1/(2**5), max=60),
           after=after_log(logger, logging.INFO),
           stop=stop_after_delay(3600),
           retry=retry_if_not_exception_type(AssertionError))
    def read_neural_data(neural_filename_ix, neural_offset_from, neural_offset_to, neural_files,
                         neural_files_basepath, n_channels, s3client, neural_sample_counts, print_exceptions: bool = True):
        """
        Read raw neural data that may cross file boundaries, either earlier (negative values) or after the end
        of the file (extending past the end of the file)
        :returns: neural data as a byte array
        """
        try:
            neural_filename_ix = np.squeeze(neural_filename_ix)
            neural_offset_from = np.squeeze(neural_offset_from)
            neural_offset_to = np.squeeze(neural_offset_to)
            neural_length = neural_offset_to - neural_offset_from

            # roll back neural_filename_ix until neural_offset_from is positive
            while neural_offset_from < 0:
                neural_filename_ix -= 1
                neural_offset_from += neural_sample_counts[neural_filename_ix]['neural_samples']
                neural_offset_to = neural_offset_from + neural_length

            # roll forward neural_filename_ix until neural_offset_from is valid
            while neural_offset_from >= neural_sample_counts[neural_filename_ix]['neural_samples']:
                neural_filename_ix += 1
                neural_offset_from -= neural_sample_counts[neural_filename_ix]['neural_samples']
                neural_offset_to = neural_offset_from + neural_length

            assert neural_filename_ix >= 0
            bytes_per_neural_sample = 2 * n_channels
            bytes_header = 8
            neural_data = b''
            next_neural_filename_ix = neural_filename_ix

            bytes_from = neural_offset_from * bytes_per_neural_sample + bytes_header
            bytes_to = neural_offset_to * bytes_per_neural_sample + bytes_header
            bytes_length = bytes_to - bytes_from

            while len(neural_data) // bytes_per_neural_sample < neural_offset_to - neural_offset_from:
                neural_filename = neural_files[next_neural_filename_ix]

                if neural_files_basepath.startswith('s3://'):
                    url = urllib.parse.urlparse(neural_files_basepath)
                    bucket = url.netloc
                    key = os.path.join(url.path[1:], neural_filename)
                    resp = s3client.get_object(Bucket=bucket, Key=key, Range='bytes={}-{}'.format(bytes_from, bytes_to - 1))
                    neural_data += resp['Body'].read()
                else:
                    with smart_open.open(os.path.join(neural_files_basepath, neural_filename), 'rb') as f:
                        f.seek(bytes_from)
                        neural_data += f.read(bytes_to - bytes_from)

                next_neural_filename_ix += 1
                bytes_from = bytes_header
                bytes_to = bytes_from + bytes_length - len(neural_data)

            return neural_data
        except Exception as e:
            if print_exceptions:
                traceback.print_exc()
                for frame in reversed(inspect.trace()):
                    print('Exception local variables:\n', frame[0].f_locals)
            raise e

    @staticmethod
    def permute_neural_data_sample(neural_data_channels_last: np.ndarray):
        assert isinstance(neural_data_channels_last, np.ndarray)
        assert len(neural_data_channels_last.shape) == 2

        neural_data = np.zeros_like(neural_data_channels_last)

        for c in range(neural_data_channels_last.shape[1]):
            neural_data[:, c] = np.random.permutation(neural_data_channels_last[:, c])

        # if neural_data.shape[0] > 1:
        #     assert not np.all(neural_data == neural_data_channels_last) and np.sum(neural_data) == np.sum(neural_data_channels_last)  # todo remove sanity check in production use.

        return neural_data


class DatasetSleepStateTrain(DatasetSleepStateBase):
    """ Sleep state model dataset for training (random sampling) """

    def __init__(self, labels_file: (str, tuple), sample_width_before: int, sample_width_after: int, n_channels: int,
                 batch_size: int, prefetch: int, data_echo_factor: int, prefetch_to_gpu: bool,
                 hp_max_predict_time_fps: int, include_modules: str,
                 neural_files_basepath: str, include_channels: (tuple, list) = None,
                 include_neural_files: (tuple, list) = None, exclude_neural_files: (tuple, list) = None,
                 # include_video_files: (tuple, list) = None, exclude_video_files: (tuple, list) = None,
                 filter_samples_near_state_change_sec: int = None, n_workers: int = 1,
                 test_video_files: (tuple, list) = None, exclude_test_video_files: bool = True,
                 video_fps: int = 15, use_single_channel_data: int = -1, permute_samples: bool = False, **_):

        super().__init__(labels_file=labels_file,
                         sample_width_before=sample_width_before,
                         sample_width_after=sample_width_after,
                         n_channels=n_channels,
                         batch_size=batch_size,
                         prefetch=prefetch,
                         prefetch_to_gpu=prefetch_to_gpu,
                         hp_max_predict_time_fps=hp_max_predict_time_fps,
                         include_modules=include_modules,
                         neural_files_basepath=neural_files_basepath,
                         include_channels=include_channels,
                         include_neural_files=include_neural_files,
                         exclude_neural_files=exclude_neural_files,
                         filter_samples_near_state_change_sec=filter_samples_near_state_change_sec,
                         n_workers=n_workers,
                         test_video_files=test_video_files,
                         exclude_test_video_files=exclude_test_video_files,
                         video_fps=video_fps,
                         use_single_channel_data=use_single_channel_data,
                         permute_samples=permute_samples)

        self.data_echo_factor = data_echo_factor

    @staticmethod
    def draw_sample(counter, subproc_properties):
        """
        Draw a single sample, returning an integer index into self.labels_matrix/self.computed_matrix

        Uses counter to coordinate across multiple workers so each produces the next element from the same shuffle.

        Rules for balanced sampling:
            - Within each objective we need balanced labels
              (ex: wake|nrem|rem, ma|not_ma, pa|not_pa, time-uniformly-sampled)
            - We need to balance the number of samples across each objective

        :param: counter a unique id >= 0 indicating which sample to produce.
        :param: subproc_properties is the deserialized version of subproc_properties passed to the init method,
            this is the way to pass information to draw_sample from the main process (a labels matrix for example).
        :return: a dictionary sample matching the values in self.output_shapes and self.data_types
        """
        # randomly choose an objective to sample using the same random permutation across multiple
        # worker processes and indexed by counter
        include_modules = subproc_properties.include_modules.split(',')
        objective = choice(counter, include_modules)

        if objective == 'wnr' or objective == 'ma' or objective == 'pa':
            label_groups = \
                [subproc_properties.ixs.all_wake, subproc_properties.ixs.all_nrem, subproc_properties.ixs.all_rem] if objective == 'wnr' else \
                [subproc_properties.ixs.all_ma, subproc_properties.ixs.not_ma] if objective == 'ma' else \
                [subproc_properties.ixs.all_wake_passive, subproc_properties.ixs.all_wake_active] if objective == 'pa' else None

            # randomly choose a class to sample
            ixs_selected = choice(counter, label_groups)

            # randomly choose a sample from the class
            ix = choice(counter, ixs_selected)

            sample_ix = IxTarget(ix, objective == 'wnr', objective == 'ma', objective == 'pa', 0, None)

        elif objective == 'time':
            time_as_sampled = np.random.default_rng().uniform(-1.0, 1.0)
            labels_time = subproc_properties.labels_computed['label_time']
            sort_ixs = subproc_properties.labels_computed['label_time_sort_ixs']
            ix_time = common_utils.arg_nearest(labels_time, time_as_sampled, sorter=sort_ixs)

            sample_ix = IxTarget(ix_time, False, False, False, True, time_as_sampled)

        else:
            raise ValueError('{} not a recognized module.'.format(objective))

        return DatasetSleepStateTrain._read_sample(sample_ix, subproc_properties)

    @staticmethod
    def _read_sample(sample_ix: IxTarget, subproc_properties):
        """
        Reads a sample from the (remote) filesystem and prepares the sample

        Note: avoid concurrent.futures.ThreadPoolExecutor due to issue:
            https://stackoverflow.com/questions/49992329/the-workers-in-threadpoolexecutor-is-not-really-daemon
        threads won't exit until finished processing due to the join() at exit.

        :param sample_ix: IxTarget object with index into labels_matrix, targets, and time_as_sampled
        :return: string reference to shared memory, use common_utils.read_shared_memory and release_shared_memory
        """
        try:
            # lazy init s3client because it can't be serialized and passed between processes
            if subproc_properties.s3client is None:
                subproc_properties.s3client = boto3.client('s3', endpoint_url=os.environ['ENDPOINT_URL'])

            subproc = subproc_properties  # per-process variables initialized by process pool
            sample = subproc.labels_matrix[sample_ix.ix]
            sample_computed = subproc.labels_computed[sample_ix.ix]

            neural_data = DatasetSleepStateBase.read_neural_data(
                neural_filename_ix=sample['neural_filename_ix'],
                neural_offset_from=int(sample['neural_offset']) - subproc.sample_width_before,
                neural_offset_to=int(sample['neural_offset']) + subproc.sample_width_after,
                neural_files=subproc.neural_files,
                neural_files_basepath=subproc.neural_files_basepath,
                n_channels=subproc.n_channels,
                s3client=subproc.s3client,
                neural_sample_counts=subproc.neural_sample_counts
            )

            neural_data_int16 = np.frombuffer(neural_data, dtype=np.int16).reshape(
                (subproc.n_channels, subproc.sample_width_after + subproc.sample_width_before), order='F'
            )

            # filter by include_channels and permute to channels last
            neural_data_channel_filtered = neural_data_int16 if subproc.include_channels is None \
                else neural_data_int16[subproc.include_channels, :]

            neural_data_channels_last = np.moveaxis(neural_data_channel_filtered, 0, 1)

            # If the option permute_samples is set, the sample will be permuted on per-channel basis
            if subproc_properties.permute_samples is True:
                neural_data_channels_last = DatasetSleepStateBase.permute_neural_data_sample(neural_data_channels_last)

            # save sample to shared memory (common_utils.create_shared_memory)
            # return string reference to shared memory
            sample = {
                'sleep_states': sample['sleep_state'],
                'label_time': sample_computed['label_time'],
                'label_time_as_sampled': sample_ix.time_as_sampled if sample_ix.time_as_sampled is not None else 0,
                'next_wake_state': sample['next_wake_state'],
                'next_nrem_state': sample['next_nrem_state'],
                'next_rem_state': sample['next_rem_state'],
                'last_wake_state': sample['last_wake_state'],
                'last_nrem_state': sample['last_nrem_state'],
                'last_rem_state': sample['last_rem_state'],
                'target_wnr': sample_ix.target_wnr,
                'target_ma': sample_ix.target_ma,
                'target_pa': sample_ix.target_pa,
                'target_time': sample_ix.target_time,
                'neural_data': neural_data_channels_last,
                'neural_length': subproc.sample_width_after + subproc.sample_width_before,
                'neural_data_offsets': 0,
                'video_frame_offsets': sample['video_frame_offset'],
                'neural_offsets': sample['neural_offset'],
                'video_filename_ixs': sample['video_filename_ix'],
                'video_filenames': subproc_properties.video_files[sample['video_filename_ix']],
                'neural_filename_ixs': sample['neural_filename_ix'],
                'neural_filenames': subproc_properties.neural_files[sample['neural_filename_ix']],
                'sample_ix': sample_ix.ix
            }

            return sample

        except Exception as e:
            traceback.print_exc()
            raise e

    def as_dataset(self):
        ds = super().as_dataset()
        ds = ds.flat_map(lambda tensor: tf.data.Dataset.from_tensors(tensor).repeat(self.data_echo_factor))
        return ds

    def output_shapes(self, unspecified_dim='np'):
        output_s = {
            'sleep_states': (),
            'label_time': (),
            'label_time_as_sampled': (),
            'next_wake_state': (),
            'next_nrem_state': (),
            'next_rem_state': (),
            'last_wake_state': (),
            'last_nrem_state': (),
            'last_rem_state': (),
            'target_wnr': (),
            'target_ma': (),
            'target_pa': (),
            'target_time': (),
            'neural_length': (),
            'neural_data_offsets': (),
            'neural_data': (
                self.sample_width_before + self.sample_width_after,
                self.n_channels if self.include_channels is None else len(self.include_channels),
            ),
            'video_frame_offsets': (),
            'neural_offsets': (),
            'video_filename_ixs': (),
            'video_filenames': (),
            'neural_filename_ixs': (),
            'neural_filenames': (),
            'sample_ix': (),
        }
        return output_s


class DatasetSleepStateEvaluate(DatasetSleepStateBase):
    """ Sleep state model dataset for evaluation (sequential sampling with multi-sample sequence for IO efficiency) """

    @staticmethod
    def draw_sample(counter, subproc_properties):
        """
        Returns a sample as one (or more, but typically one) contiguous sample which spans multiple samples.
        The data in these samples overlaps so the neural data is passed in without the need for overlap this way,
        saving the overlapping IO.

        Typically one sample is generated, but it if there is a discontinuity in the neural data then a batch
        of 2+ samples may be generated at the discontinuity.

        Note that the batch_size property is used differently in DatasetSleepStateEvaluate than it is in
        DatasetSleepStateTrain. Here it determines the number of samples to group into a single contiguous segment
        of neural data.

        :param counter: a global incrementing counter indexing the data, multiple processes can be called with different
            counter values and will yield results consistently across processes.
        :param subproc_properties: any properties passed to the constructors subproc_properties that need
            to be passed to the subprocesses.
        :return:
        """
        assert subproc_properties.batch_size >= 1

        # lazy init s3client because it can't be serialized and passed between processes
        if subproc_properties.s3client is None:
            subproc_properties.s3client = boto3.client('s3', endpoint_url=os.environ['ENDPOINT_URL'])

        # Compute the number of batches of samples available in the full labels_matrix dataset
        n_batch_partitions = math.ceil(len(subproc_properties.labels_matrix) / subproc_properties.batch_size)

        # counter indexes the batches of labels_matrix samples
        # When counter is out of range we return None to indicate end of data stream
        if counter >= n_batch_partitions:
            return None  # no more samples to process in sequence

        samples_batch = np.array_split(subproc_properties.labels_matrix, n_batch_partitions)[counter]
        samples_calculated_batch = np.array_split(subproc_properties.labels_computed, n_batch_partitions)[counter]

        # compute a set of 1 or more disjoint neural data ranges that provide data for all samples, it is expected
        # that most samples will contain overlapping neural data segments and thus use one neural_data sample.
        block_neural_data_ranges, relative_offset_per_sample = DatasetSleepStateEvaluate.compute_overlapping_neural_segments(
            samples=samples_batch,
            sample_width_before=subproc_properties.sample_width_before,
            sample_width_after=subproc_properties.sample_width_after,
            neural_sample_counts=subproc_properties.neural_sample_counts,
        )

        neural_data_blocks = [
            DatasetSleepStateEvaluate.read_neural_data(
                neural_filename_ix=bndr[0],
                neural_offset_from=bndr[1],
                neural_offset_to=bndr[2],
                neural_files=subproc_properties.neural_files,
                neural_files_basepath=subproc_properties.neural_files_basepath,
                n_channels=subproc_properties.n_channels,
                s3client=subproc_properties.s3client,
                neural_sample_counts=subproc_properties.neural_sample_counts,
            )
            for bndr in block_neural_data_ranges
        ]
        dynamic_sample_size = sum([bndr[2] - bndr[1] for bndr in block_neural_data_ranges])
        assert dynamic_sample_size is not None and dynamic_sample_size > 0, \
            f'Error computing sample shape.\n' \
            f'\tdynamic_sample_size: {dynamic_sample_size}\n' \
            f'\tblock_neural_data_ranges\n{block_neural_data_ranges}\n' \
            f'\tcounter {counter}\n' \
            f'\tn_batch_partitions {n_batch_partitions}\n' \
            f'\tsamples_batch {samples_batch}\n' \
            f'\trelative_offset_per_sample {relative_offset_per_sample}\n'

        neural_data = b''.join(neural_data_blocks)
        neural_data_npy = np.frombuffer(neural_data, dtype=np.int16)
        neural_data_npy = np.reshape(neural_data_npy, (-1, dynamic_sample_size), order='F').T

        # filter by include_channels
        if subproc_properties.include_channels is not None:
            neural_data_npy = neural_data_npy[:, subproc_properties.include_channels]

        # If the option permute_samples is set, the sample will be permuted on per-channel basis
        if subproc_properties.permute_samples is True:
            neural_data_npy = DatasetSleepStateBase.permute_neural_data_sample(neural_data_npy)

        sample = {
            'sleep_states': samples_batch['sleep_state']
                .reshape(subproc_properties.output_shapes['sleep_states']),
            'label_time': samples_calculated_batch['label_time']
                .reshape(subproc_properties.output_shapes['label_time']),
            'label_time_as_sampled': samples_calculated_batch['label_time_as_sampled']
                .reshape(subproc_properties.output_shapes['label_time']),
            'next_wake_state': samples_batch['next_wake_state']
                .reshape(subproc_properties.output_shapes['next_wake_state']),
            'next_nrem_state': samples_batch['next_nrem_state']
                .reshape(subproc_properties.output_shapes['next_nrem_state']),
            'next_rem_state': samples_batch['next_rem_state']
                .reshape(subproc_properties.output_shapes['next_rem_state']),
            'last_wake_state': samples_batch['last_wake_state']
                .reshape(subproc_properties.output_shapes['last_wake_state']),
            'last_nrem_state': samples_batch['last_nrem_state']
                .reshape(subproc_properties.output_shapes['last_nrem_state']),
            'last_rem_state': samples_batch['last_rem_state']
                .reshape(subproc_properties.output_shapes['last_rem_state']),
            'target_wnr': np.array([], dtype=np.bool),
            'target_ma': np.array([], dtype=np.bool),
            'target_pa': np.array([], dtype=np.bool),
            'target_time': np.array([], dtype=np.bool),
            'neural_length': np.array(subproc_properties.sample_width_before + subproc_properties.sample_width_after, dtype=np.int32),
            'neural_data_offsets': relative_offset_per_sample['relative_offset'].astype(np.int32)
                .reshape(subproc_properties.output_shapes['neural_data_offsets']),
            'neural_data': neural_data_npy,
            'video_frame_offsets': samples_batch['video_frame_offset'].astype(np.int64)
                .reshape(subproc_properties.output_shapes['video_frame_offsets']),
            'neural_offsets': samples_batch['neural_offset'].astype(np.int64)
                .reshape(subproc_properties.output_shapes['neural_offsets']),
            'video_filename_ixs': samples_batch['video_filename_ix']
                .reshape(subproc_properties.output_shapes['video_filename_ixs']),
            'video_filenames': [subproc_properties.video_files[sample['video_filename_ix']] for sample in samples_batch],
            'neural_filename_ixs': samples_batch['neural_filename_ix']
                .reshape(subproc_properties.output_shapes['neural_filename_ixs']),
            'neural_filenames': [subproc_properties.neural_files[sample['neural_filename_ix']] for sample in samples_batch],
            'sample_ix': np.arange(subproc_properties.batch_size * counter, subproc_properties.batch_size * counter + len(samples_batch))
        }

        return sample

    @staticmethod
    def compute_overlapping_neural_segments(samples, sample_width_before, sample_width_after, neural_sample_counts):
        """
        Aggregates overlapping segments of neural_data into optimal contiguous chunks

        Worked example:

        sample_width = 1000

        # get neural samples per file
        (11, 33000)
        (12, 29000)

        # get filename_ix and neural offset
        (11, 32401)
        (11, 32707)
        (11, 32806)
        (12, 105)
        (12, 1801)

        # convert to absolute ranges by subtracting the sample width, may cross file boundaries
        (11, 31401, 32401)
        (11, 31707, 32707)
        (11, 31806, 32806)
        (12, -995, 105)
        (12, 1701, 1801)

        # normalize to lowest filename_ix (33000 + (-995)) and (33000 + 105) and (33000 + 1701) and (33000 + 1801)
        (11, 31401, 32401)
        (11, 31707, 32707)
        (11, 31806, 32806)
        (11, 32005, 33105)
        (11, 34701, 34801)

        # order by starting neural_offset
        (11, 31401, 32401)
        (11, 31707, 32707)
        (11, 31806, 32806)
        (11, 32005, 33105)
        (11, 34701, 34801)

        # eliminate overlap
        (11, 31401, 32401)
        (11, 31401, 32707)
        (11, 31401, 32806)
        (11, 31401, 33105)
        (11, 31401, 34801) -> (11, 31401, 34801)

        # Compute relative offset per sample
        (0, 306, 405, 604, 3300)

        return: [(11, 31401, 34801, 0, 306, 405, 604, 3300), ...]
                   f  start   end   relative offset per sample

        When multiple segments exist just concatenate them and update the relative offset per sample
        """

        # convert to absolute ranges by subtracting the sample width, may cross file boundaries
        neural_sample_range = recfunctions.merge_arrays(
            (samples['neural_filename_ix'],
             samples['neural_offset'] - sample_width_before,
             samples['neural_offset'] + sample_width_after),
            flatten=True, usemask=False,
        ).astype(dtype=[('neural_filename_ix', '<i4'),
                        ('start_neural_offset', '<i8'),
                        ('end_neural_offset', '<i8')])

        # normalize to lowest filename_ix
        lowest_filename_ix = np.min(neural_sample_range['neural_filename_ix'])
        for ix in np.where(neural_sample_range['neural_filename_ix'] > lowest_filename_ix)[0]:
            delta_filename_ix = neural_sample_range[ix]['neural_filename_ix'] - lowest_filename_ix
            delta_neural_samples = np.sum(neural_sample_counts['neural_samples'][lowest_filename_ix:delta_filename_ix])
            neural_sample_range[ix]['start_neural_offset'] += delta_neural_samples
            neural_sample_range[ix]['end_neural_offset'] += delta_neural_samples
            neural_sample_range[ix]['neural_filename_ix'] = lowest_filename_ix

        # order by starting neural_offset (this should already be the case, but the sanity check here is good)
        sort_ixs = np.argsort(neural_sample_range['start_neural_offset'])
        neural_sample_range = neural_sample_range[sort_ixs]

        # eliminate overlap
        block_begin = neural_sample_range[0][1]
        block_end = neural_sample_range[0][2]
        block_neural_data_ranges = []
        for nsr in neural_sample_range[1:]:
            # Check if there is overlap
            if max(nsr['start_neural_offset'], block_begin) < min(nsr['end_neural_offset'], block_end):
                block_begin = min(nsr['start_neural_offset'], block_begin)
                block_end = max(nsr['end_neural_offset'], block_end)
            else:
                block_neural_data_ranges.append((lowest_filename_ix, block_begin, block_end))
                block_begin = nsr[1]
                block_end = nsr[2]
        block_neural_data_ranges.append((lowest_filename_ix, block_begin, block_end))
        block_neural_data_ranges = np.array(block_neural_data_ranges, dtype=[('neural_filename_ix', '<i4'),
                                                                             ('start_neural_offset', '<i8'),
                                                                             ('end_neural_offset', '<i8')])

        # compute relative offset per sample
        relative_offset_per_sample = np.zeros(shape=len(neural_sample_range), dtype=[('relative_offset', '<i8')])
        for i, nsr in enumerate(neural_sample_range):
            # select the appropriate block
            block_ix = np.where(
                (nsr['start_neural_offset'] >= block_neural_data_ranges['start_neural_offset']) &
                (nsr['end_neural_offset'] <= block_neural_data_ranges['end_neural_offset'])
            )[0][0]
            relative_offset = \
                nsr['start_neural_offset'] - \
                block_neural_data_ranges[block_ix]['start_neural_offset'] + \
                np.sum((block_neural_data_ranges['end_neural_offset'] -
                        block_neural_data_ranges['start_neural_offset'])[:block_ix])
            relative_offset_per_sample[i] = relative_offset

        return block_neural_data_ranges, relative_offset_per_sample

    def output_shapes(self, unspecified_dim='np'):
        """
        :param unspecified_dim: 'np' or 'tf' for numpy or tensorflow style unspecified dimension, -1 or None
        :return:
        """
        assert unspecified_dim in ['np', 'tf']
        unspecified_dim = -1 if unspecified_dim == 'np' else None
        output_s = {
            'sleep_states': (unspecified_dim,),
            'label_time': (unspecified_dim,),
            'label_time_as_sampled': (unspecified_dim,),
            'next_wake_state': (unspecified_dim,),
            'next_nrem_state': (unspecified_dim,),
            'next_rem_state': (unspecified_dim,),
            'last_wake_state': (unspecified_dim,),
            'last_nrem_state': (unspecified_dim,),
            'last_rem_state': (unspecified_dim,),
            'target_wnr': (unspecified_dim,),
            'target_ma': (unspecified_dim,),
            'target_pa': (unspecified_dim,),
            'target_time': (unspecified_dim,),
            'neural_length': (),
            'neural_data_offsets': (unspecified_dim,),
            'neural_data': (unspecified_dim,
                            self.n_channels if self.include_channels is None else len(self.include_channels)),
            'video_frame_offsets': (unspecified_dim,),
            'neural_offsets': (unspecified_dim,),
            'video_filename_ixs': (unspecified_dim,),
            'video_filenames': (unspecified_dim,),
            'neural_filename_ixs': (unspecified_dim,),
            'neural_filenames': (unspecified_dim,),
            'sample_ix': (unspecified_dim,),
        }
        return output_s

    def as_dataset(self):
        """ Override as_dataset because batch_size is used in a different way at Evaluate time. """
        ds = tf.data.Dataset.from_generator(
            generator=self.sample_generator,
            output_types=self.output_types,
            output_shapes=self.output_shapes('tf'),
        )
        ds = ds.prefetch(buffer_size=2)
        if self.prefetch_to_gpu:
            ds = ds.apply(tf.data.experimental.prefetch_to_device(
                device='gpu:0' if tf.test.is_gpu_available(cuda_only=True) else 'cpu', buffer_size=1
            ))
        return ds


def get_neural_file_sample_counts(neural_files: (list, tuple), neural_files_basepath: str, n_channels: int):
    # noinspection PyBroadException
    def get_length(nf, basepath, chan):
        # file_size = smart_open.open(os.path.join(basepath, nf), mode='rb').size()
        try:
            filename = os.path.join(basepath, nf)
            file_size = common_utils.file_size(filename)
        except ClientError as e:
            # noinspection PyUnboundLocalVariable
            print(f'Exception getting file size for: {filename if "filename" in locals() else None}')
            raise e

        assert (file_size - 8) % (chan * 2) == 0, \
            f'Unexpected file size for {filename}. Got file size {file_size}, ' \
            f'which isn\'t an even multiple of channels, validation criteria: ({file_size} - 8) % ({chan} * 2) == 0'
        return (file_size - 8) // (chan * 2)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    n = len(neural_files)
    neural_files_sample_lengths = list(executor.map(
        get_length, neural_files, [neural_files_basepath] * n, [n_channels] * n
    ))

    neural_files_sample_lengths = np.array(neural_files_sample_lengths, dtype=[('neural_samples', '<i8')])

    return neural_files_sample_lengths


def interleave_samples(cluster_ix):
    """
    Compute interleaved clusters, producing a list of cluster IDs that
    interleave optimally based on cluster_bincounts

    :return: indexes locations into cluster_ix which produce a random permutation of values in cluster interleaved order
    """
    cluster_bincounts = np.bincount(cluster_ix)
    csum = int(np.sum(cluster_bincounts))

    interleave = np.zeros(shape=csum, dtype=np.int32)
    frequencies = csum / cluster_bincounts
    x = [(f, 0, i) for i, f in enumerate(frequencies)]
    heapq.heapify(x)
    for i in range(csum):
        priority, _, cluster_n = heapq.heappop(x)
        interleave[i] = cluster_n
        heapq.heappush(x, (priority + frequencies[cluster_n], i, cluster_n))

    # Loop over each cluster number and assign the frame_ids to the right cluster and performing random permutation
    result = np.zeros(shape=cluster_ix.sz, dtype=np.uint32)
    rng = np.arange(cluster_ix.sz[0])
    for cluster_n in range(len(cluster_bincounts)):
        result[interleave == cluster_n] = np.random.permutation(rng[cluster_ix == cluster_n])

    return result


def img_decode_png(sample, resolution_height_width):
    img_decoded = tf.io.decode_png(sample['frame'])
    img_decoded.set_shape((resolution_height_width[0], resolution_height_width[1], 3))
    return {'frame': img_decoded}


def parse_args_with_dataset_config(parser):
    """
    Runs parser.parse_args() plus applies DATASET_NAME.cfg overrides. This is a general utility function.
    :return: vars(parsed_args)
    """
    parsed_args = parser.parse_args()

    if hasattr(parsed_args, 'dataset_config'):
        # Optionally parse values from the DATASET_NAME.cfg config file
        cfg = parse_dataset_config(parsed_args.dataset_config, as_cli_arguments=True)
        ix = next((i for i, x in enumerate(sys.argv[1:]) if x.startswith('-')), None)
        unhyphened_args = sys.argv[1:ix + 1]
        hyphened_args = sys.argv[1 + ix:]
        parsed_args, _ = parser.parse_known_args(unhyphened_args + cfg + hyphened_args)
        print('> parsing dataset config: ', parsed_args)
        if 'dataset_config' in parsed_args:
            del parsed_args.dataset_config

    return vars(parsed_args)


def parse_dataset_config(dataset_config: str, as_cli_arguments=True):
    """
    Reads a DATASET_NAME.cfg file and returns the parameters as a dictionary.

    :param dataset_config: A dataset_config file path
    :param as_cli_arguments: Returns keys in the form '--arg_name' for use in argparse.
    :return: A dictionary of parameters in the .cfg file
    """
    print('Reading dataset_config: {}'.format(dataset_config))

    if dataset_config is None:
        cfg_args = {}
    else:
        if dataset_config.startswith('s3://'):
            o = urllib.parse.urlparse(dataset_config)
            filesystem = S3FS(bucket_name=o.netloc, dir_path='/', endpoint_url=os.environ['ENDPOINT_URL'], strict=False)
            filepath = o.path
        else:
            path, filepath = os.path.split(dataset_config)
            filesystem = OSFS(path)

        with filesystem.open(filepath) as f:
            config_parser = configparser.ConfigParser()
            config_parser.read_file(f)
            cfg_args = dict(config_parser.items('CONFIG'))

    if as_cli_arguments:
        cfg_args = list(itertools.chain.from_iterable([('--{}'.format(k), v) for k, v in cfg_args.items()]))

    return cfg_args


@lru_cache(maxsize=1)
def load_labels_file(labels_file: str):
    """ Loads the labels file from local or S3. Cached operation because it may be used in multiple places. """
    with smart_open.open(labels_file, 'rb') as f:
        print('Loading labels file: {}'.format(labels_file))
        b = f.read()  # more efficient to read in one chunk
        labels_npz = np.load(io.BytesIO(b))
        labels_matrix = labels_npz['labels_matrix']
        video_files = labels_npz['video_files']
        neural_files = labels_npz['neural_files']

    return labels_matrix, video_files, neural_files


def parse_n_channels(n_channels: int, filename: str):
    """ Returns n_channels if it's an int already, or parses n_channels from a filename. """
    if n_channels is None:
        n_channels = int(re.findall(r'_(\d*)_Channels', filename)[0])
    return n_channels


def assert_mutually_exclusive(valid_dtypes: (list, tuple), **kwargs):
    """
    Validates that two arguments are mutually exclusive, use keyword parameters so assert messages look good.

    Example usage:
        validate_mutually_exclusive(include_video_files=valueA, exclude_video_files=valueB)

    :param valid_dtypes: an iterable of valid dtypes for the arguments
    :param kwargs: pass two keyword named arguments, see example.
    """
    assert len(kwargs) == 2, 'Incorrect function usage, see function documentation.'
    ((argA, valA), (argB, valB)) = kwargs.items()
    assert not (valA is not None and valB is not None), '{} is mutually exclusive with {}'.format(argA, argB)

    for f in itertools.chain(valA or (), valB or ()):
        assert type(f) in valid_dtypes, '{} and {} must contain dtypes {}.'.format(valA, valB, valid_dtypes)


def include_exclude_mask(items: (tuple, list), data: np.ndarray,
                         include_list: (tuple, list), exclude_list: (str, tuple, list)):
    """
    :param items: A list of items, for example neural_filenames|indexes or video_filenames|indexes (str or int)
    :param data: An ndarray of item indexes
    :param include_list: a list of indexes or strings to only include (mutually exclusive with exclude_list)
    :param exclude_list: a list of indexes or strings to only exclude (mutually exclusive with include_list)
    :return:
    """
    assert isinstance(items, (tuple, list, np.ndarray)) and isinstance(include_list, (tuple, list, type(None))) \
           and isinstance(exclude_list, (tuple, list, type(None))) and isinstance(data, np.ndarray)

    # Validate include|exclude lists are contained in items
    if include_list and isinstance(include_list[0], str):
        assert np.all(np.isin(include_list or (), items)), \
            'Given items {} were not found in the data which contains {}' \
                .format(include_list, items)
    if exclude_list and isinstance(exclude_list[0], str):
        assert np.all(np.isin(exclude_list or (), items)), \
            'Given items {} were not found in the data which contains {}' \
                .format(exclude_list, items)

    # Resolve names to indexes
    include_list_ixs = include_list if include_list is None or isinstance(include_list[0], int) else \
        np.array([np.argmax([strname == i for i in items]) for strname in include_list])
    exclude_list_ixs = exclude_list if exclude_list is None or isinstance(exclude_list[0], int) else \
        np.array([np.argmax([strname == i for i in items]) for strname in exclude_list])

    # Set mask for inclusions and exclusions
    mask = np.isin(data, include_list_ixs) if include_list is not None else ~np.isin(data, exclude_list_ixs)

    return mask


def compute_label_ixs(labels_matrix):
    """ Label subsets of interest which we only want to compute once """
    ixs = types.SimpleNamespace()
    ixs.all_wake_active = np.where(labels_matrix['sleep_state'] == 1)[0]
    ixs.all_wake_passive = np.where(labels_matrix['sleep_state'] == 5)[0]
    ixs.all_wake = np.union1d(ixs.all_wake_active, ixs.all_wake_passive)
    ixs.all_nrem = np.where(labels_matrix['sleep_state'] == 2)[0]
    ixs.all_rem = np.where(labels_matrix['sleep_state'] == 3)[0]

    ixs.all_ma = np.where(labels_matrix['sleep_state'] == 4)[0]
    ixs.not_ma = np.where(labels_matrix['sleep_state'] != 4)[0]

    print(
        'Samples per class: all_ma {}, all_nrem {}, all_rem {}, all_wake {}, '
        'all_wake_active {}, all_wake_passive {}, not_ma {}'.format(
            len(ixs.all_ma), len(ixs.all_nrem), len(ixs.all_rem), len(ixs.all_wake),
            len(ixs.all_wake_active), len(ixs.all_wake_passive), len(ixs.not_ma)
        )
    )

    return ixs


def choice(counter, a):
    """
    Deterministic + balanced choice with deterministic randomness based on counter value.
    Different processes will produce the same permutations depending on counter.
    choice() returns a in permuted order as long as counter is incremented by 1 each call.
    """
    # noinspection PyArgumentList
    rg = np.random.Generator(np.random.PCG64(seed=counter // len(a)))
    ix = rg.permutation(len(a))[counter % len(a)]
    return a[ix]
