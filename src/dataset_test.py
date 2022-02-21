""" Unittests for the DatasetHengenlabMiceFrameGenerator class. """
import unittest
from unittest import mock
import numpy as np
import dao
import tensorflow.compat.v1 as tf
import dataset
import io
from PIL import Image
import fs
import fs.tempfs
from deprecated import deprecated
import os
import multiprocessing
import threading
import queue
import itertools
import hanging_threads
import time
import boto3
import common_py_utils.common_utils as common_py_utils
from common_py_utils.test_utils import BaseTestCase
from deprecated import deprecated
import types
import logging
import sys
tf.disable_eager_execution()


logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

mock_read_sample_n_calls = 0  # global variable for test_sample_generator_with_data_echoing


class DatasetPreprocessingTest(BaseTestCase):
    def test_tfdataset_pipeline(self):
        ds = tf.data.Dataset.from_tensor_slices(
            {'frame': np.ones(shape=(2, 480, 640, 3), dtype=np.uint8)}
        )
        dspp = dataset.DatasetImgPreprocessing(ds=ds)
        iterator = iter(dspp.as_dataset())
        val = next(iterator)
        self.assertTrue(val['frame'].sz == dataset.IMG_SHAPE_PADDED)

    def test_ndarray_pipeline(self):
        dspp = dataset.DatasetImgPreprocessing(ds=None)

        x = np.ones(shape=(480, 640, 3), dtype=np.uint8)
        val = dspp.preprocessing_pipeline(x)

        self.assertTrue(val['frame'].sz == dataset.IMG_SHAPE_PADDED)

    def test_img_resize_and_pad(self):
        dspp = dataset.DatasetImgPreprocessing()

        test_img = {'frame': np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 200}

        img_resized = dspp.img_resize(sample=test_img)['frame']

        self.assertTrue(img_resized.sz == (120, 160, 3))

    def test_img_rescale(self):
        x = {'frame': np.arange(256, dtype=np.uint8)}
        result = dataset.DatasetImgPreprocessing.img_rescale(x)['frame']

        self.assertTrue(result.dtype == np.float32)
        self.assertAlmostEqual(np.min(result), -1.0)
        self.assertAlmostEqual(np.max(result), 1.0)

    def test_img_pad(self):
        dspp = dataset.DatasetImgPreprocessing()
        test_img = {'frame': np.zeros(shape=(120, 160, 3), dtype=np.uint8) + 200}
        result_tensor = dspp.img_pad(test_img)['frame']
        self.assertTrue(result_tensor.sz == (128, 160, 3))

    def test_img_black_text_area(self):
        dspp = dataset.DatasetImgPreprocessing()
        test_img = {'frame': np.zeros(shape=(128, 160, 3), dtype=np.uint8) + 200}
        result = dspp.img_black_text_area(test_img)['frame']
        self.assertTrue(result.sz == (128, 160, 3))  # just sanity check here

    def test_dataset_preprocessing(self):
        img_pil = Image.fromarray(128 * np.ones(shape=(480, 640, 3), dtype=np.uint8))
        img_npy = np.array(img_pil)

        dspp = dataset.DatasetImgPreprocessing()

        result = dspp.preprocessing_pipeline(img_npy)['frame']

        self.assertTrue(np.all(result > -1.0))
        self.assertTrue(np.all(result < 1.0))
        self.assertTrue(np.all(result[5:11, 48:112, :] == 0.0))


# noinspection DuplicatedCode
class DatasetNeuralSequenceLabelsTest(BaseTestCase):
    def setUp(self) -> None:
        self.tfs = fs.tempfs.TempFS()
        dtype = [('ecube_time', '<u8'),
                 ('video_frame_global_ix', '<u4'),
                 ('video_filename_ix', '<i4'),
                 ('video_frame_offset', '<u4'),
                 ('neural_filename_ix', '<i4'),
                 ('neural_offset', '<i8'),
                 ('sleep_state', 'i1'),
                 ('dlc_label', '<f4', (6,)),
                 ('cluster_ix', '<i4'),
                 ('frame_data_offset', '<u8'),
                 ('frame_data_length', '<u4')]

        self.labels_neuralendecode = np.zeros(shape=(4,), dtype=dtype)
        self.labels_neuralendecode['ecube_time'] = [10, 20, 30, 40]
        self.labels_neuralendecode['video_frame_global_ix'] = [0, 1, 2, 3]
        self.labels_neuralendecode['video_filename_ix'] = [0, 0, 1, 1]
        self.labels_neuralendecode['video_frame_offset'] = [0, 1, 0, 1]
        self.labels_neuralendecode['neural_filename_ix'] = [0, 1, 1, 2]
        self.labels_neuralendecode['neural_offset'] = [100, 200, 300, 400]
        self.labels_neuralendecode['sleep_state'] = -1
        self.labels_neuralendecode['dlc_label'] = -1
        self.labels_neuralendecode['cluster_ix'] = 0
        self.labels_neuralendecode['frame_data_offset'] = [10, 20, 30, 40]
        self.labels_neuralendecode['frame_data_length'] = [40000, 40001, 40002, 40003]

        self.video_files = np.array(['video_file_0', 'video_file_1'])
        self.neural_files = np.array(['neural_file_0', 'neural_file_1', 'neural_file_2'])

        self.labels_neuralendecode_filename = self.tfs.getsyspath('/') + 'labels_neuralendecode_UNITTEST.npz'

        np.savez(self.labels_neuralendecode_filename,
                 labels_neuralendecode=self.labels_neuralendecode,
                 video_files=self.video_files, neural_files=self.neural_files)

        for fname in self.neural_files:
            with self.tfs.openbin(fname, mode='wb+') as f:
                f.write(os.urandom(8 + 1000))  # 64 bit timestamp, 500 samples, 1 channel, 2 bytes per data point

    def tearDown(self) -> None:
        self.tfs.close()

    def test_generate_all(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 0,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 0,
             'frame_data_offset': 10, 'frame_data_length': 40000,
             'neural_filename_ix': 0, 'neural_filenames': 'neural_file_0', 'neural_offset': 100, 'neural_length': 150,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 1,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 1,
             'frame_data_offset': 20, 'frame_data_length': 40001,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 200, 'neural_length': 150,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 2,
             'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 0,
             'frame_data_offset': 30, 'frame_data_length': 40002,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 300, 'neural_length': 150,
             'cluster_ix': 0},
            # {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 3,
            #  'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 1,
            #  'frame_data_offset': 40, 'frame_data_length': 40003,
            #  'neural_filename_ix': 2, 'neural_filenames': 'neural_file_2', 'neural_offset': 400, 'neural_length': 150,
            #  'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=0, sequence_length=1, n_channels=1, fs=150)
        gen = ds.generate()
        results = [next(gen) for _ in range(3)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_include_neural_files(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 0,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 0,
             'frame_data_offset': 10, 'frame_data_length': 40000,
             'neural_filename_ix': 0, 'neural_filenames': 'neural_file_0', 'neural_offset': 100, 'neural_length': 150,
             'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=0, sequence_length=1, n_channels=1, fs=150,
                                                 include_neural_files=['neural_file_0'])
        gen = ds.generate()
        results = [next(gen) for _ in range(1)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_exclude_neural_files(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 0,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 0,
             'frame_data_offset': 10, 'frame_data_length': 40000,
             'neural_filename_ix': 0, 'neural_filenames': 'neural_file_0', 'neural_offset': 100, 'neural_length': 150,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 1,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 1,
             'frame_data_offset': 20, 'frame_data_length': 40001,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 200, 'neural_length': 150,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 2,
             'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 0,
             'frame_data_offset': 30, 'frame_data_length': 40002,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 300, 'neural_length': 150,
             'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=0, sequence_length=1, n_channels=1, fs=150,
                                                 exclude_neural_files=['neural_file_2'])
        gen = ds.generate()
        results = [next(gen) for _ in range(3)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_include_video_files(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 0,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 0,
             'frame_data_offset': 10, 'frame_data_length': 40000,
             'neural_filename_ix': 0, 'neural_filenames': 'neural_file_0', 'neural_offset': 100, 'neural_length': 150,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 1,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 1,
             'frame_data_offset': 20, 'frame_data_length': 40001,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 200, 'neural_length': 150,
             'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=0, sequence_length=1, n_channels=1, fs=150,
                                                 include_video_files=['video_file_0'])
        gen = ds.generate()
        results = [next(gen) for _ in range(2)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_exclude_video_files(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 2,
             'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 0,
             'frame_data_offset': 30, 'frame_data_length': 40002,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 300, 'neural_length': 150,
             'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=0, sequence_length=1, n_channels=1, fs=150,
                                                 exclude_video_files=[0])
        gen = ds.generate()
        results = [next(gen) for _ in range(1)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_include_video_exclude_neural(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 1,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 1,
             'frame_data_offset': 20, 'frame_data_length': 40001,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 200, 'neural_length': 150,
             'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=0, sequence_length=1, n_channels=1, fs=150,
                                                 include_video_files=[0],
                                                 exclude_neural_files=[0])
        gen = ds.generate()
        results = [next(gen) for _ in range(1)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_filter_prefix_sample_length(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 1,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 1,
             'frame_data_offset': 20, 'frame_data_length': 40001,
             'neural_filename_ix': 0, 'neural_filenames': 'neural_file_0:neural_file_1', 'neural_offset': 450, 'neural_length': 300,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 2,
             'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 0,
             'frame_data_offset': 30, 'frame_data_length': 40002,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 300, 'neural_length': 300,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 3,
             'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 1,
             'frame_data_offset': 40, 'frame_data_length': 40003,
             'neural_filename_ix': 2, 'neural_filenames': 'neural_file_2', 'neural_offset': 400, 'neural_length': 300,
             'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=250, sequence_length=1, n_channels=1, fs=50)
        gen = ds.generate()
        results = [next(gen) for _ in range(3)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)

    def test_generate_filter_sequence_length_forward_valid(self):
        expected_results = [
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 0,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 0,
             'frame_data_offset': 10, 'frame_data_length': 40000,
             'neural_filename_ix': 0, 'neural_filenames': 'neural_file_0', 'neural_offset': 100, 'neural_length': 226,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 1,
             'video_filename_ix': 0, 'video_filename': 'video_file_0', 'video_frame_offset': 1,
             'frame_data_offset': 20, 'frame_data_length': 40001,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1', 'neural_offset': 200, 'neural_length': 226,
             'cluster_ix': 0},
            {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 2,
             'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 0,
             'frame_data_offset': 30, 'frame_data_length': 40002,
             'neural_filename_ix': 1, 'neural_filenames': 'neural_file_1:neural_file_2', 'neural_offset': 300, 'neural_length': 226,
             'cluster_ix': 0},
            # {'DATASET_NAME': 'UNITTEST', 'latent_state_row_index': 3,
            #  'video_filename_ix': 1, 'video_filename': 'video_file_1', 'video_frame_offset': 1,
            #  'frame_data_offset': 40, 'frame_data_length': 40003,
            #  'neural_filename_ix': 2, 'neural_filenames': 'neural_file_2', 'neural_offset': 400, 'neural_length': 50,
            #  'cluster_ix': 0},
        ]

        ds = dataset.DatasetNeuralSequenceLabels(labels_neuralendecode=self.labels_neuralendecode_filename,
                                                 neural_files_basepath=self.tfs.getsyspath('/'),
                                                 prefix_sample_length=1, sequence_length=3, n_channels=1, fs=75)
        gen = ds.generate()
        results = [next(gen) for _ in range(3)]

        self.assertCountEqual(results, expected_results)

        with self.assertRaises(StopIteration):
            next(gen)


class DatasetDeltaLatentStateLoaderTest(BaseTestCase):
    def setUp(self) -> None:
        self.tfs = fs.tempfs.TempFS()
        self.latent_state = np.random.rand(4, 6).astype(np.float32)
        self.latent_state_file = os.path.join(self.tfs.getsyspath('/'), 'latent_state_UNITTEST.npy')
        np.save(self.latent_state_file, self.latent_state)

    def tearDown(self) -> None:
        self.tfs.close()

    def test_generate(self):
        input_data = {
            'DATASET_NAME': b'UNITTEST',
            'latent_state_row_index': 1,
            'video_filename_ix': 0,
            'video_filename': b'video_file_0',
            'video_frame_offset': 10,
            'frame_data_offset': 11,
            'frame_data_length': 40,
            'neural_filename_ix': 0,
            'neural_filename': b'neural_file_0',
            'neural_offset': 0,
            'cluster_ix': 0,
        }
        tfds = tf.data.Dataset.from_tensors(input_data)
        dsnsls = dataset.DatasetDeltaLatentStateLoader(latent_state=self.latent_state_file, sequence_length=2)

        get_next = dsnsls.as_dataset(tfds).make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            result = sess.run(get_next)

            expected_result = {
                **input_data,
                'initial_latent_state': self.latent_state[1, :],
                'delta_latent_state': self.latent_state[2:4, :] - self.latent_state[1:3, :],
            }

            self.assertEqual(str(result), str(expected_result))

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run(get_next)


class DatasetNeuralLoaderTest(BaseTestCase):
    def setUp(self) -> None:

        self.tfs = fs.tempfs.TempFS()
        for i, filename in enumerate(['file1', 'file2', 'file3', 'file4']):
            f = self.tfs.openbin(filename, mode='wb')
            # 64 bit timestamp 'T' and 3 channels, 10 datapoints, 2 byte data '#' corresponding to file 1|2|3|4
            f.write(b'T' * 8 + str(i + 1).encode('utf-8') * (3 * 10 * 2))

        self.ds_input = tf.data.Dataset.from_tensor_slices({
            'neural_offset': [0, 5, 0, 0],
            'neural_length': [15, 10, 10, 5],
            'neural_filenames': ['file1:file2', 'file2:file3', 'file3', 'file4'],
        })

        self.sess = tf.Session()

    def tearDown(self) -> None:
        self.tfs.close()
        self.sess.close()
        tf.reset_default_graph()

    def test_read_neural_data_single_file(self):
        expected_result = np.frombuffer(b'1' * 30, dtype=np.int16).reshape((3, 5), order='F')

        dsnsrl = dataset.DatasetNeuralLoader(neural_files_basepath=self.tfs.getsyspath('/'), n_channels=3)
        result = dsnsrl.read_neural_data(0, 5, 'file1'.encode('utf-8'))

        self.assertTrue(np.all(result == expected_result))

    def test_read_neural_data_multiple_files(self):
        expected_result = np.frombuffer(b'1' * 30 + b'2' * 30, dtype=np.int16).reshape((3, 10), order='F')

        dsnsrl = dataset.DatasetNeuralLoader(neural_files_basepath=self.tfs.getsyspath('/'), n_channels=3)
        result = dsnsrl.read_neural_data(5, 10, 'file1:file2'.encode('utf-8'))

        self.assertTrue(np.all(result == expected_result))

    def test_generate(self):
        dsnsrl = dataset.DatasetNeuralLoader(neural_files_basepath=self.tfs.getsyspath('/'), n_channels=3)
        ds = dsnsrl.as_dataset(self.ds_input)
        get_next = ds.make_one_shot_iterator().get_next()

        expected_result = [
            {'neural_offset': 0, 'neural_length': 15, 'neural_filenames': b'file1:file2',
             'neural_data': np.frombuffer(b'1' * 60 + b'2' * 30, dtype=np.int16).reshape((3, 15), order='F')},
            {'neural_offset': 5, 'neural_length': 10, 'neural_filenames': b'file2:file3',
             'neural_data': np.frombuffer(b'2' * 30 + b'3' * 30, dtype=np.int16).reshape((3, 10), order='F')},
            {'neural_offset': 0, 'neural_length': 10, 'neural_filenames': b'file3',
             'neural_data': np.frombuffer(b'3' * 60, dtype=np.int16).reshape((3, 10), order='F')},
            {'neural_offset': 0, 'neural_length':  5, 'neural_filenames': b'file4',
             'neural_data': np.frombuffer(b'4' * 30, dtype=np.int16).reshape((3, 5), order='F')},
        ]

        result = [self.sess.run(get_next) for _ in range(4)]

        with self.assertRaises(tf.errors.OutOfRangeError):
            self.sess.run(get_next)

        self.assert_complex_equal(result, expected_result)

    def test_other_values_pass_through(self):
        dsnsrl = dataset.DatasetNeuralLoader(neural_files_basepath=self.tfs.getsyspath('/'), n_channels=3)
        ds = dsnsrl.as_dataset(self.ds_input)
        ds = ds.map(lambda tensors: {**tensors, 'additional_value': 42})
        get_next = ds.make_one_shot_iterator().get_next()

        expected_result = [
            {'neural_offset': 0, 'neural_length': 15, 'neural_filenames': b'file1:file2', 'additional_value': 42,
             'neural_data': np.frombuffer(b'1' * 60 + b'2' * 30, dtype=np.int16).reshape((3, 15), order='F')},
            {'neural_offset': 5, 'neural_length': 10, 'neural_filenames': b'file2:file3', 'additional_value': 42,
             'neural_data': np.frombuffer(b'2' * 30 + b'3' * 30, dtype=np.int16).reshape((3, 10), order='F')},
            {'neural_offset': 0, 'neural_length': 10, 'neural_filenames': b'file3', 'additional_value': 42,
             'neural_data': np.frombuffer(b'3' * 60, dtype=np.int16).reshape((3, 10), order='F')},
            {'neural_offset': 0, 'neural_length':  5, 'neural_filenames': b'file4', 'additional_value': 42,
             'neural_data': np.frombuffer(b'4' * 30, dtype=np.int16).reshape((3, 5), order='F')},

        ]

        result = [self.sess.run(get_next) for _ in range(4)]

        with self.assertRaises(tf.errors.OutOfRangeError):
            self.sess.run(get_next)

        self.assert_complex_equal(result, expected_result)


class DatasetSleepStateTrainTest(BaseTestCase):
    def setUp(self) -> None:
        self.tfs, self.basepath, self.labels_filename, self.neural_files, self.video_files, self.neural_sample_counts = \
            create_label_file_and_neural_data_5state()
        z = np.load(self.labels_filename)
        self.labels_matrix = z['labels_matrix']
        self.video_files = z['video_files']
        self.neural_files = z['neural_files']

    def test_sample_generator_wo_data_echoing(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=3, sample_width_after=6, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,pa,time',  # ma samples get filtered out by these params
            neural_files_basepath=self.tfs.getsyspath('/'), n_workers=1,
        )

        g = dsst.sample_generator()
        sample = next(g)

        self.assertTrue(isinstance(sample, dict))  # simple sanity check, most of the detailed checks in other tests
        self.assertTrue(sample['neural_data'].shape == (9, 1))

    def test_filter_labels_matrix(self):
        """ Test that filter_labels_matrix filters samples before/after that don't have sufficient neural data. """
        labels_matrix_filtered = dataset.DatasetSleepStateTrain.filter_labels_matrix(
            labels_matrix=self.labels_matrix, video_files=self.video_files, neural_files=self.neural_files,
            neural_sample_counts=self.neural_sample_counts, sample_width_before=3, sample_width_after=6,
            include_neural_files=None, exclude_neural_files=None, include_video_files=None, exclude_video_files=None,
        )
        self.assertTrue(np.all(labels_matrix_filtered['sleep_state'] == np.array([1, 2, 5])))

    def test_sample_generator_multiprocess(self):
        """ This test is attempting to reproduce a concurrency issue. """
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,ma,pa,time',
            neural_files_basepath=self.tfs.getsyspath('/'), n_workers=2,
        )

        g = dsst.sample_generator()
        next(g)
        next(g)

        self.assertTrue(True)  # completion only check

    def test_as_dataset(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,ma,pa,time',
            neural_files_basepath=self.tfs.getsyspath('/'), n_workers=1,
        )

        ds = dsst.as_dataset()
        get_next_tensor = tf.data.make_one_shot_iterator(ds).get_next()

        with tf.Session() as sess:
            result = sess.run(get_next_tensor)

        self.assertTrue(isinstance(result, dict))  # basic sanity check

    def test_draw_samples_wnr_ma_pa_time(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,ma,pa,time',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        result = [dsst.draw_sample(i, dsst.subproc_properties) for i in range(48)]

        n_wnr = np.sum([r['target_wnr'] for r in result])
        n_ma = np.sum([r['target_ma'] for r in result])
        n_pa = np.sum([r['target_pa'] for r in result])
        n_time = np.sum([r['target_time'] for r in result])

        self.assertTrue(n_wnr == n_ma == n_pa == n_time)

    def test_draw_samples_wnr(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        result = [dsst.draw_sample(i, dsst.subproc_properties) for i in range(12)]

        n_wnr = np.sum([r['target_wnr'] for r in result])
        ixs = [r['sample_ix'] for r in result]

        self.assertTrue(n_wnr == 12)
        self.assertCountEqual(ixs, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4])

    def test_draw_samples_ma(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='ma',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        result = [dsst.draw_sample(i, dsst.subproc_properties) for i in range(12)]

        n_ma = np.sum([r['target_ma'] for r in result])
        ixs = [r['sample_ix'] for r in result]

        self.assertTrue(n_ma == 12)
        self.assertCountEqual(np.unique(ixs), [0, 1, 2, 3, 4])
        self.assertEqual(6, sum([1 if ix == 3 else 0 for ix in ixs]))

    def test_draw_samples_pa(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='pa',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        result = [dsst.draw_sample(i, dsst.subproc_properties) for i in range(12)]

        n_pa = np.sum([r['target_pa'] for r in result])
        ixs = [r['sample_ix'] for r in result]

        self.assertTrue(n_pa == 12)
        self.assertCountEqual(np.unique(ixs), [0, 4])
        self.assertEqual(6, sum([1 if ix == 0 else 0 for ix in ixs]))
        self.assertEqual(6, sum([1 if ix == 4 else 0 for ix in ixs]))

    def test_draw_samples_time(self):
        dsst = dataset.DatasetSleepStateTrain(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, data_echo_factor=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='time',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        result = [dsst.draw_sample(i, dsst.subproc_properties) for i in range(12 * 1000)]

        np.random.seed(42)
        n_time = np.sum([r['target_time'] for r in result])
        ixs = [r['sample_ix'] for r in result]
        uniq, counts = np.unique(ixs, return_counts=True)

        # only a basic sanity check, full validation is a bit more complicated than is worthwhile
        self.assertTrue(n_time == 12 * 1000)
        self.assertCountEqual(uniq, [0, 1, 2, 4])  # sample ix 3 excluded because it's time value is -2


class DatasetSleepStateEvaluateTest(BaseTestCase):
    def setUp(self) -> None:
        self.tfs, self.basepath, self.labels_filename, self.neural_files, self.video_files, self.neural_sample_counts = \
            create_label_file_and_neural_data()
        z = np.load(self.labels_filename)
        self.labels_matrix = z['labels_matrix']
        self.video_files = z['video_files']
        self.neural_files = z['neural_files']

    def test_draw_samples_sequential(self):
        dsst = dataset.DatasetSleepStateEvaluate(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=2, prefetch=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,ma,pa,time',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        result = dsst.draw_sample(counter=0, subproc_properties=dsst.subproc_properties)

        self.assertTrue(len(result['sleep_states']) == 2)

    def test_bug_eab40xyf(self):
        """ This is an online test with specific file dependencies. """
        dsst = dataset.DatasetSleepStateEvaluate(
            labels_file='../dataset/EAB40-XYF/Labels/labels_sleepstate_v2.1_EAB40-XYF.npz',
            # labels_file='s3://hengenlab/EAB40-XYF/Labels/labels_sleepstate_v2.1_EAB40-XYF.npz',
            sample_width_before=65536,
            sample_width_after=0,
            n_channels=256,
            batch_size=20,
            prefetch=1,
            prefetch_to_gpu=False,
            hp_max_predict_time_fps=900,
            include_modules='wnr',
            neural_files_basepath='s3://hengenlab/EAB40-XYF/Neural_Data/',
            exclude_neural_files=[
                'Headstages_256_Channels_int16_2019-04-02_11-54-54_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-03_08-20-01_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-04_17-10-11_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-04_17-15-11_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-05_16-15-18_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-06_08-45-24_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-06_13-20-25_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-06_18-25-27_CORRECTED.bin',
                'Headstages_256_Channels_int16_2019-04-08_04-55-29_CORRECTED.bin',
           ],
            test_video_files=['e3v8100-20190402T1350-1450.mp4']
        )

        for counter in [209]:
            result = dsst.draw_sample(counter=counter, subproc_properties=dsst.subproc_properties)
            self.assertTrue(len(result['sleep_states']) == 20)   # sanity check, bug will evoke an exception

    def test_draw_all_samples_batch_size_1(self):
        dsst = dataset.DatasetSleepStateEvaluate(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=1, prefetch=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,ma,pa,time',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        samples = []
        while (sample:=dsst.draw_sample(counter=len(samples), subproc_properties=dsst.subproc_properties)) is not None:
            if len(samples) > len(self.labels_matrix):
                raise IndexError
            samples.append(sample)

        result = [s['sleep_states'][0] for s in samples]
        expected = self.labels_matrix['sleep_state'].tolist()
        self.assert_complex_equal(result, expected)

    def test_draw_all_samples_batch_size_2(self):
        dsst = dataset.DatasetSleepStateEvaluate(
            labels_file=self.labels_filename, sample_width_before=1, sample_width_after=0, n_channels=1,
            batch_size=2, prefetch=1, prefetch_to_gpu=False,
            hp_max_predict_time_fps=900, include_modules='wnr,ma,pa,time',
            neural_files_basepath=self.tfs.getsyspath('/'),
        )

        samples = []
        while (sample:=dsst.draw_sample(counter=len(samples), subproc_properties=dsst.subproc_properties)) is not None:
            if len(samples) > len(self.labels_matrix):
                raise IndexError
            samples.append(sample)

        self.assertEqual(2, len(samples))
        result = list(itertools.chain.from_iterable([s['sleep_states'] for s in samples]))
        expected = self.labels_matrix['sleep_state'].tolist()
        self.assert_complex_equal(result, expected)

    # These tests need to be updated for sample_width_before and sample_width_after to be used
    # They are removed for now because they're not working and haven't been updated
    # They may be re-instated later if this function proves problematic.
    # def test_compute_overlapping_neural_segments(self):
    #     dtype = [('neural_samples', np.uint32)]
    #
    #     result = dataset.DatasetSleepStateEvaluate.compute_overlapping_neural_segments(
    #         self.labels_matrix, 6, np.array([10, 10], self.neural_sample_counts, dtype=dtype)
    #     )
    #
    #     expected_result = (
    #         np.array([(0, 2, 14)], dtype=[('neural_filename_ix', '<i4'),
    #                                       ('start_neural_offset', '<i8'),
    #                                       ('end_neural_offset', '<i8')]),
    #         np.array([0, 2, 6], dtype=[('relative_offset', '<i8')])
    #     )
    #
    #     self.assertTrue(np.all(expected_result[0] == result[0]))
    #     self.assertTrue(np.all(expected_result[1] == result[1]))
    #
    # def test_compute_overlapping_neural_segments_two_blocks(self):
    #     dtype = [('neural_samples', np.uint32)]
    #
    #     labels_matrix = np.hstack((
    #         self.labels_matrix,
    #         np.array((-1, 1, -2, -2, -2, -2, -2, -2, 1, 0, 2, 10), dtype=self.labels_matrix.dtype)
    #     ))
    #     neural_sample_counts = np.array([10, 10, 10], dtype=dtype)
    #
    #     result = dataset.DatasetSleepStateEvaluate.compute_overlapping_neural_segments(
    #         labels_matrix, 6, neural_sample_counts,
    #     )
    #
    #     expected_result = (
    #         np.array([(0, 2, 14), (0, 24, 30)], dtype=[('neural_filename_ix', '<i4'),
    #                                                    ('start_neural_offset', '<i8'),
    #                                                    ('end_neural_offset', '<i8')]),
    #         np.array([0, 2, 6, 12], dtype=[('relative_offset', '<i8')])
    #     )
    #
    #     self.assertTrue(np.all(expected_result[0] == result[0]))
    #     self.assertTrue(np.all(expected_result[1] == result[1]))
    #
    # def test_compute_overlapping_neural_segments_three_blocks(self):
    #     dtype = [('neural_samples', np.uint32)]
    #
    #     labels_matrix = np.hstack((
    #         self.labels_matrix,
    #         np.array((-1, 1, -2, -2, -2, -2, -2, -2, 1, 0, 2, 10), dtype=self.labels_matrix.dtype),
    #         np.array((-1, 1, -2, -2, -2, -2, -2, -2, 1, 0, 3, 8), dtype=self.labels_matrix.dtype),
    #     ))
    #     neural_sample_counts = np.array([10, 10, 10, 10], dtype=dtype)
    #
    #     result = dataset.DatasetSleepStateEvaluate.compute_overlapping_neural_segments(
    #         labels_matrix, 6, neural_sample_counts,
    #     )
    #
    #     expected_result = (
    #         np.array([(0, 2, 14),
    #                   (0, 24, 30),
    #                   (0, 32, 38)],
    #                  dtype=[('neural_filename_ix', '<i4'),
    #                         ('start_neural_offset', '<i8'),
    #                         ('end_neural_offset', '<i8')]),
    #         np.array([0, 2, 6, 12, 18], dtype=[('relative_offset', '<i8')])
    #     )
    #
    #     self.assertTrue(np.all(expected_result[0] == result[0]))
    #     self.assertTrue(np.all(expected_result[1] == result[1]))

    def test_read_sample(self):
        neural_data = dataset.DatasetSleepStateBase.read_neural_data(
            neural_filename_ix=0, neural_offset_from=5, neural_offset_to=10,
            neural_files=self.neural_files, neural_files_basepath=self.tfs.getsyspath('/'),
            n_channels=1, s3client=None, neural_sample_counts=self.neural_sample_counts,
        )

        self.assertEqual(b'a' * 10, neural_data)

    def test_read_sample_crossing_file_boundaries(self):
        neural_data = dataset.DatasetSleepStateBase.read_neural_data(
            neural_filename_ix=0, neural_offset_from=5, neural_offset_to=15, neural_files=self.neural_files,
            neural_files_basepath=self.tfs.getsyspath('/'), n_channels=1, neural_sample_counts=self.neural_sample_counts, s3client=None
        )

        self.assertEqual(b'a' * 10 + b'b' * 10, neural_data)

    def test_read_sample_negative(self):
        neural_data = dataset.DatasetSleepStateBase.read_neural_data(
            neural_filename_ix=1, neural_offset_from=-6, neural_offset_to=0,
            neural_files=self.neural_files, neural_files_basepath=self.tfs.getsyspath('/'),
            n_channels=1, s3client=None, neural_sample_counts=self.neural_sample_counts,
        )

        self.assertEqual(b'a' * 12, neural_data)

    # @unittest.skip('online test')
    def test_read_sample_online_s3(self):
        s3client = boto3.client('s3', endpoint_url=os.environ['ENDPOINT_URL'])

        neural_data = dataset.DatasetSleepStateBase.read_neural_data(
            neural_filename_ix=0, neural_offset_from=5, neural_offset_to=15,
            neural_files=['Headstages_256_Channels_int16_2019-03-30_09-33-35.bin'],
            neural_files_basepath='s3://braingeneersdev/dfparks/test/', n_channels=2,
            neural_sample_counts=[7500000], s3client=s3client
        )

        self.assertEqual(len(neural_data), 40)


class DatasetBaseSharedMemoryTest(BaseTestCase):
    sample = {'unittest': np.array([1, 2, 3]), 'other_scalar': 42, 'other_list': ['a', 'b', 'c']}

    class ImplDatasetBaseSharedMemory(dataset.DatasetBaseSharedMemory):
        def draw_sample(self, counter, properties):
            assert properties['unittest'] == 'validation'
            return DatasetBaseSharedMemoryTest.sample

    def test_sample_generator(self):
        dbsm = self.ImplDatasetBaseSharedMemory(n_workers=1, subproc_properties={'unittest': 'validation'})
        g = dbsm.sample_generator()
        sample = next(g)
        self.assert_complex_equal(DatasetBaseSharedMemoryTest.sample, sample)


labels_dtype = [
    ('activity', np.int8),
    ('sleep_state', np.int8),
    ('next_wake_state', np.int64),
    ('next_nrem_state', np.int64),
    ('next_rem_state', np.int64),
    ('last_wake_state', np.int64),
    ('last_nrem_state', np.int64),
    ('last_rem_state', np.int64),
    ('video_filename_ix', np.int32),
    ('video_frame_offset', np.int32),
    ('neural_filename_ix', np.int32),
    ('neural_offset', np.int64)
]


# noinspection DuplicatedCode
def create_label_file_and_neural_data():
    """ Writes a labels_file_unittest.npy to a TempFS """
    tfs = fs.tempfs.TempFS()
    basepath = tfs.getsyspath('/')
    labels_filename = '{}/labels_sleepstate_unittest.npz'.format(basepath)

    # Create labels_file
    labels_matrix = np.empty(shape=(3,), dtype=labels_dtype)
    labels_matrix['activity'] =           [-1, -1, -1]
    labels_matrix['sleep_state'] =        [ 1,  2,  3]
    labels_matrix['next_wake_state'] =    [-1, -2, -2]
    labels_matrix['next_nrem_state'] =    [-2, -1,  1]
    labels_matrix['next_rem_state'] =     [-2, -2, -1]
    labels_matrix['last_wake_state'] =    [-1,  1,  2]
    labels_matrix['last_nrem_state'] =    [-2, -1,  1]
    labels_matrix['last_rem_state'] =     [-2, -2, -1]
    labels_matrix['video_filename_ix'] =  [ 0,  0,  1]
    labels_matrix['video_frame_offset'] = [ 0,  0,  0]
    labels_matrix['neural_filename_ix'] = [ 0,  1,  1]
    labels_matrix['neural_offset'] =      [ 8,  0,  4]
    video_files = np.array(['video_file1.mp4', 'video_file2.mp4'])
    neural_files = np.array(['neural_file1.bin', 'neural_file2.bin'])
    np.savez(labels_filename, labels_matrix=labels_matrix, video_files=video_files, neural_files=neural_files)
    # Each test neural file is 20 bytes long
    with tfs.openbin(neural_files[0], 'w') as f0, tfs.openbin(neural_files[1], 'w') as f1:
        f0.write(b'T' * 8)   # timestamp, 8 bytes
        f0.write(b'a' * 20)  # data, 10 samples, 20 bytes
        f1.write(b'T' * 8)   # timestamp, 8 bytes
        f1.write(b'b' * 20)  # data, 10 samples, 20 bytes
    with tfs.openbin(neural_files[0] + '.tail', 'w') as tail:
        tail.write(b'x' * 4)
    with tfs.openbin(neural_files[1] + '.head', 'w') as head:
        head.write(b'T' * 8 + b'x' * 4)
    neural_sample_counts = np.array([10, 10], dtype=[('neural_samples', '<i8')])

    return tfs, basepath, labels_filename, neural_files, video_files, neural_sample_counts


# noinspection DuplicatedCode
def create_label_file_and_neural_data_5state():
    """ Writes a labels_file_unittest.npy to a TempFS """
    tfs = fs.tempfs.TempFS()
    basepath = tfs.getsyspath('/')
    labels_filename = '{}/labels_sleepstate_unittest.npz'.format(basepath)

    # Create labels_file
    labels_matrix = np.empty(shape=(5,), dtype=labels_dtype)
    labels_matrix['activity'] =           [-1, -1, -1, -1, -1]
    labels_matrix['sleep_state'] =        [ 1,  2,  3,  4,  5]
    labels_matrix['next_wake_state'] =    [-1,  3,  2,  1, -1]
    labels_matrix['next_nrem_state'] =    [ 1, -1, -2, -2, -2]
    labels_matrix['next_rem_state'] =     [ 2,  1, -1, -2, -2]
    labels_matrix['last_wake_state'] =    [-1,  1,  2,  3, -1]
    labels_matrix['last_nrem_state'] =    [-2, -1,  1,  2,  3]
    labels_matrix['last_rem_state'] =     [-2, -2, -1,  1,  2]
    labels_matrix['video_filename_ix'] =  [ 0,  0,  1,  1,  0]
    labels_matrix['video_frame_offset'] = [ 0,  0,  0,  0,  0]
    labels_matrix['neural_filename_ix'] = [ 0,  1,  1,  0,  0]
    labels_matrix['neural_offset'] =      [ 8,  0,  4,  2,  4]
    video_files = np.array(['video_file1.mp4', 'video_file2.mp4'])
    neural_files = np.array(['neural_file1.bin', 'neural_file2.bin'])
    np.savez(labels_filename, labels_matrix=labels_matrix, video_files=video_files, neural_files=neural_files)
    # Each test neural file is 20 bytes long
    with tfs.openbin(neural_files[0], 'w') as f0, tfs.openbin(neural_files[1], 'w') as f1:
        f0.write(b'T' * 8)   # timestamp, 8 bytes
        f0.write(b'a' * 20)  # data, 10 samples, 20 bytes
        f1.write(b'T' * 8)   # timestamp, 8 bytes
        f1.write(b'b' * 20)  # data, 10 samples, 20 bytes
    with tfs.openbin(neural_files[0] + '.tail', 'w') as tail:
        tail.write(b'x' * 4)
    with tfs.openbin(neural_files[1] + '.head', 'w') as head:
        head.write(b'T' * 8 + b'x' * 4)
    neural_sample_counts = np.array([10, 10], dtype=[('neural_samples', '<i8')])

    return tfs, basepath, labels_filename, neural_files, video_files, neural_sample_counts
