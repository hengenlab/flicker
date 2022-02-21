import warnings
warnings.filterwarnings('ignore')
import comet_ml
import tensorflow.compat.v1 as tf
import numpy as np
import dataset
import common_py_utils.growing_epochs as growing_epochs
import common_py_utils.model_base as model_base
from common_py_utils.common_utils import SimpleNamespace
from common_py_utils.model_base import FetchOp
import math
import typing
import re
import itertools
import time
import sklearn
import io
import uuid
import zipfile
import queue
import threading
import os
import smart_open

tf.disable_eager_execution()


# def arg_parser():
#     parser = argparse.ArgumentParser(description='Hengenlab mouse data neural RNN model')
#     subparsers = parser.add_subparsers(dest='command')
#
#     #
#     # Common parameters
#     #
#     parser_common = argparse.ArgumentParser(add_help=False)
#
#     parser_common.add_argument(
#         '--labels_file', required=True, type=str,
#         help='The sleep state labels file, S3 or local filesystem. '
#              'Example: ../dataset/SCF05/labels_sleepstate_SCF05.npz'
#     )
#     parser_common.add_argument(
#         '--load_model', type=str, required=False,
#         help='Model checkpoint to load, ex: checkpoints/2019-08-22-03h-52m-28s-step016000-loss0.399.index'
#     )
#     parser_common.add_argument(
#         '--dataset_config', required=False, type=str,
#         help='A dataset .cfg file which defines specific details and quirks of the dataset. Any values specified '
#              'in this file override the command line parameters.'
#     )
#     parser_common.add_argument(
#         '--neural_files_basepath', required=True, default=None, type=str,
#         help='The base path (s3 or filesystem) to the neural data files. Example: s3://hengenlab/SCF05/Neural_Data/'
#     )
#     parser_common.add_argument(
#         '--include_channels', required=False, type=str,
#         help='Specify one or more (comma separated or range) channels to include from training (excluding all others), '
#              'this can be optionally read from the config file. Example: --include_channels 0:64,192:256'
#     )
#     parser_common.add_argument(
#         '--n_channels', required=False, type=int, default=None,
#         help='Number of channels of the input data, value will be parsed automatically from the filename if None.'
#     )
#     parser_common.add_argument(
#         '--sample_width', required=False, type=int, default=65536,
#         help='Size of input data.'
#     )
#     parser_common.add_argument(
#         '--corrupt_neural_files', required=False, type=str,
#         help='Comma separated list of neural filenames (no path) to skip (due to data corruption, for example).'
#     )
#     parser_common.add_argument(
#         '--include_modules', required=False, default=None, type=str,
#         help='A CSV string of modules to include, example: "wnr,ma,pa,time" or "wnr", or "wnr,time"'
#     )
#     parser_common.add_argument(
#         '--hp_max_predict_time_fps', required=False, default=900, type=int,
#         help='Hyperparameter max time in frames-per-second (15 fps in hengenlab data) '
#              'to predict sleep|wake state changes.'
#     )
#
#     #
#     # train subparser
#     #
#     parser_train = subparsers.add_parser(
#         'train', help='Model training', parents=[parser_common],
#     )
#     parser_train.add_argument(
#         '--training_steps', default=10, type=int, help='Number of training steps.'
#     )
#     parser_train.add_argument(
#         '--checkpoint', nargs='?', default=None, const='../checkpoints/dev/', type=str,
#         help='Checkpoints directory, defaults to: ../checkpoints/dev/ if no argument is given.'
#     )
#     parser_train.add_argument(
#         '--checkpoint_final', default=None, type=str,
#         help='Final (last) checkpoint path (if different than the path for --checkpoint).'
#     )
#     parser_train.add_argument(
#         '--disable_comet', default=False, action='store_true',
#         help='Disables comet output.'
#     )
#     parser_train.add_argument(
#         '--test_video_files', required=False, type=str, default=None,
#         help='Video files to generate video for, comma separated list, may be configured in DATASET_NAME.cfg, '
#              'unset will generate all video files found in --video_file_basepath.'
#     )
#     parser_train.add_argument(
#         '--batch_size', required=False, default=8, type=int,
#     )
#     parser_train.add_argument(
#         '--testeval_on_checkpoint', default=False, action='store_true',
#         help='A flag that submits a test eval job to kubernetes every time a checkpoint is created.'
#     )
#     parser_train.add_argument(
#         '--n_workers', required=False, type=int, default=16,
#         help='Number of worker processes to read samples, defaults to 16.'
#     )
#
#     #
#     # common prediction parser
#     #
#     parser_predict_common = argparse.ArgumentParser(add_help=False, parents=[parser_common])
#
#     parser_predict_common.add_argument(
#         '--include_video_files', type=str, nargs='+', default=None,
#         help='Space separated list of video filenames or indexes. '
#              'Example: --include_video_files e3v8100-20190330T1428-1528.mp4 e3v8100-20190330T0928-1028.mp4'
#     )
#     parser_predict_common.add_argument(
#         '--limit_samples', required=False, type=int, default=None,
#         help='Limits the number of samples drawn for testing purposes.'
#     )
#     parser_predict_common.add_argument(
#         '--local_cache_size', required=False, type=int, default=50e+9,
#         help='Max size of local samples cache. Defaults to a value appropriate for training on GPU.'
#     )
#     parser_predict_common.add_argument(
#         '--local_cache_workers', required=False, type=int, default=4,
#         help='Number of data sampling worker processes, default to 4.'
#     )
#
#     #
#     # predict subparser
#     #
#     parser_predict = subparsers.add_parser(
#         'predict', help='Model prediction', parents=[parser_predict_common],
#     )
#     parser_predict.add_argument(
#         '--output_file', required=False, type=str, default=None,
#         help='Writes predictions to the specified CSV file, None prints to stdout.'
#     )
#
#     #
#     # testeval subparser
#     #
#     parser_testeval = subparsers.add_parser(
#         'testeval', help='Test set evaluation and reporting during training', parents=[parser_predict_common],
#     )
#     parser_testeval.add_argument(
#         '--experiment_key', required=True, type=str, default=None,
#         help='Comet.ml key to report results to.'
#     )
#
#     #
#     # summary statistics subparser
#     #
#     parser_stats = subparsers.add_parser('generate_summary_statistics')
#
#     parser_stats.add_argument(
#         '--predictions_filename', type=str, required=True,
#         help='Input path/filename to the predictions_DATASET_NAME.csv, maybe be local or S3.'
#     )
#     parser_stats.add_argument(
#         '--output_filename', type=str, required=False, default=None,
#         help='Path/filename of output file, defaults to path/summary_stats_DATASET_NAME.csv.'
#     )
#     parser_stats.add_argument(
#         '--quiet', action='store_false', required=False, default=True, dest='display',
#         help='Flag, do not display results to STDOUT (defaults to writing to STDOUT and file.'
#     )
#     parser_stats.add_argument(
#         '--test_video_files', type=str, required=False,
#         help='Comma separated list of video file names that were held out as test and will be marked as such.'
#     )
#     parser_stats.add_argument(
#         '--dataset_config', required=False, type=str,
#         help='A dataset .cfg file which defines specific details and quirks of the dataset. Any values specified '
#              'in this file override the command line parameters.'
#     )
#
#     #
#     # Finalize
#     #
#     result = dataset.parse_args_with_dataset_config(parser)
#     print('Command line args', result)
#     return result
#
#
# noinspection DuplicatedCode
class ModelSleepState(model_base.ModelBase):
    def __init__(self,
                 session: tf.Session = None,
                 device: str = None,
                 checkpoint_period_min: float = 60,
                 scope: str = '',
                 dataset_obj=None,
                 # HYPER-PARAMETERS
                 include_modules:str = 'wnr,ma,pa,time',
                 batch_size: int = 1,
                 sample_width: int = None,
                 include_channels: (tuple, list) = None,
                 build_loss: bool = True,
                 hp_fc_wnr_n_neurons: int = 150,  # Last fully connected layer for loss 1 size
                 hp_fc_ma_n_neurons: int = 150,  # Same for loss 2
                 hp_fc_pa_n_neurons: int = 150,  # Same for loss 3
                 hp_fc_time_n_neurons: int = 150,  # Same for loss 4
                 hp_scale_loss_wnr: float = 1.0,  # scale WAKE|NREM|REM loss
                 hp_scale_loss_ma: float = 1.0,  # scale MA loss
                 hp_scale_loss_pa: float = 1.0,  # scale PA loss
                 hp_scale_loss_time: float = 1.0,  # scale TIME loss
                 hp_max_predict_time_fps: int = None  # max time in frames-per-second used to compute time_label
                 ) -> None:

        super().__init__(session, device, checkpoint_period_min, scope)

        self.sample_width = sample_width
        self.n_channels = dataset_obj.n_channels
        self.build_loss = build_loss
        self.hp_fc_wnr_n_neurons = hp_fc_wnr_n_neurons
        self.hp_fc_ma_n_neurons = hp_fc_ma_n_neurons
        self.hp_fc_pa_n_neurons = hp_fc_pa_n_neurons
        self.hp_fc_time_n_neurons = hp_fc_time_n_neurons
        self.hp_scale_loss_wnr = hp_scale_loss_wnr
        self.hp_scale_loss_ma = hp_scale_loss_ma
        self.hp_scale_loss_pa = hp_scale_loss_pa
        self.hp_scale_loss_time = hp_scale_loss_time
        self.hp_max_predict_time_fps = hp_max_predict_time_fps
        self.include_modules = include_modules
        self.batch_size = batch_size

        if include_channels is None or include_channels == 'all':
            self.include_channels = None
        elif isinstance(include_channels, str):
            self.include_channels = [int(x) for x in include_channels.split(',')]
        elif isinstance(include_channels, int):
            self.include_channels = [include_channels]
        elif isinstance(include_channels, (list, tuple)):
            self.include_channels = include_channels
        else:
            raise TypeError('include_channels is of type {}, expected None, "all", int, list or tuple.'.format(type(include_channels)))

    @property
    def fetch_ops(self):
        metrics_update_frequency = 200
        loss_update_frequency = 10

        ops = [
            # Training
            FetchOp(name='train_op', tensor=self.ops.train_op,
                    update_frequency=1, is_reported=False),
            FetchOp(name='metrics_update_op', tensor=self.ops.metrics_update_op,
                    update_frequency=1, is_reported=False),
            FetchOp(name='loss', tensor=self.ops.loss,
                    update_frequency=loss_update_frequency, is_reported=True),
            # Metrics
            FetchOp(name='reset_local_variables', tensor=self.ops.reset_local_variables,
                    update_frequency=metrics_update_frequency, is_reported=False),
            FetchOp(name='loss_wnr', tensor=self.ops.loss_wnr,
                    update_frequency=metrics_update_frequency, is_reported=True),
            FetchOp(name='accuracy_wnr', tensor=self.ops.accuracy_wnr,
                    update_frequency=metrics_update_frequency, is_reported=True),
            FetchOp(name='error_rate_wnr_long_term_state', tensor=self.ops.error_rate_wnr_long_term_state,
                    update_frequency=metrics_update_frequency, is_reported=True),
            FetchOp(name='error_rate_wnr_short_term_state', tensor=self.ops.error_rate_wnr_short_term_state,
                    update_frequency=metrics_update_frequency, is_reported=True),
        ]
        return ops

    def status_message(self, out: SimpleNamespace):
        """ Generates an appropriate status message for this model. """
        return "TODO"

    def build_model(self, input_tensors: dict, build_loss: bool = True) -> None:
        assert self.hp_max_predict_time_fps is not None, 'Required parameter'

        with tf.variable_scope(self.scope), tf.device(self.device):
            # Input tensors
            input_neural = input_tensors['neural_data']
            neural_data_offsets = input_tensors['neural_data_offsets']
            sleep_state = input_tensors['sleep_states']
            label_time = input_tensors['label_time']
            label_time_as_sampled = input_tensors['label_time_as_sampled'] if 'label_time_as_sampled' in input_tensors else None
            target_wnr = input_tensors['target_wnr'] if 'target_wnr' in input_tensors else None
            target_ma = input_tensors['target_ma'] if 'target_ma' in input_tensors else None
            target_pa = input_tensors['target_pa'] if 'target_pa' in input_tensors else None
            target_time = input_tensors['target_time'] if 'target_time' in input_tensors else None

            assert self.include_channels is None or (isinstance(self.include_channels, (tuple, list)) and len(self.include_channels) > 0)
            assert self.sample_width is not None
            assert self.n_channels is not None
            assert isinstance(input_neural, str) or input_neural.shape.dims is not None, 'Input tensor shape is not known.'

            self.ops.global_step = tf.train.get_or_create_global_step()
            self.ops.reset_local_variables = tf.local_variables_initializer()

            # Expand sequential batches into a batch of dense, split samples
            # Example: neural_data = [10,20,30,40,50], neural_data_offsets = [0, 2], self.sample_width = 3
            #   x <- [[10,20,30],
            #         [30,40,50]]
            if len(input_neural.shape) == 2:
                gather_indices = tf.range(0, self.sample_width)
                gather_indices = tf.expand_dims(gather_indices, axis=0)
                gather_indices = tf.tile(gather_indices, [tf.shape(neural_data_offsets)[0], 1])
                gather_indices = tf.add(gather_indices, tf.expand_dims(neural_data_offsets, axis=1))
                x = tf.gather(input_neural, gather_indices, name='gather_neural_from_indices')
                x.set_shape((None, self.sample_width, len(self.include_channels) if self.include_channels else self.n_channels))
            else:
                x = input_neural
                x.set_shape((None, self.sample_width, x.shape.as_list()[-1]))

            self.ops.x = x
            print('DEBUG> ', x.shape)

            x = 0.001 * (tf.cast(x, dtype=tf.float32) if x.dtype == tf.int16 else x)
            assert x.dtype == tf.float32

            self.ops.cnn_output = self.build_cnn(x)

            x = tf.squeeze(self.ops.cnn_output, axis=1)  # the data dimension (axis 1) is removed.

            # Build outputs and subbatches for each output
            # Some outputs may not be trained on depending on options, but they will be constructed anyway
            # so the same prediction operations work in any case.
            self.build_outputs(self.ops, x, self.hp_fc_wnr_n_neurons, self.hp_fc_ma_n_neurons,
                               self.hp_fc_pa_n_neurons, self.hp_fc_time_n_neurons)

            if build_loss:
                self.build_losses(self.ops, sleep_state, label_time_as_sampled,
                                  target_wnr, target_ma, target_pa, target_time,
                                  self.hp_scale_loss_wnr, self.hp_scale_loss_ma, self.hp_scale_loss_pa, self.hp_scale_loss_time)

                self.build_metrics(self.ops, label_time, self.hp_max_predict_time_fps)

                self.ops.train_op = self.build_optimizer(self.ops.loss)

    @staticmethod
    def build_outputs(ops, x, hp_fc_wnr_n_neurons, hp_fc_ma_n_neurons, hp_fc_pa_n_neurons, hp_fc_time_n_neurons):
        """
        Sleep state labels:
          1  WAKE - ACTIVE
          2  NREM
          3  REM
          4  MICRO-AROUSAL    (only applicable iff WAKE==1)
          5  WAKE - PASSIVE
        """
        # V4 model outputs
        # fc_wnr = tf.keras.layers.Dense(units=hp_fc_wnr_n_neurons)(x)
        # ops.output_wnr = tf.keras.layers.Dense(units=3)(fc_wnr)  # WAKE|NREM|REM classification
        #
        # fc_ma = tf.keras.layers.Dense(units=hp_fc_ma_n_neurons)(x)
        # ops.output_ma = tf.keras.layers.Dense(units=1)(fc_ma)  # MA (micro-arousal)
        #
        # fc_pa = tf.keras.layers.Dense(units=hp_fc_pa_n_neurons)(x)
        # ops.output_pa = tf.keras.layers.Dense(units=1)(fc_pa)  # PA (passive-active wake)
        #
        # fc_time = tf.keras.layers.Dense(units=hp_fc_time_n_neurons)(x)
        # fc_time = tf.math.tanh(fc_time)
        # ops.output_time = tf.keras.layers.Dense(units=1)(fc_time)

        glorot_uniform_initializer = tf.initializers.glorot_uniform()
        zeros_initializer = tf.zeros_initializer()

        with tf.variable_scope('vars_output'):
            fc_wnr_weights = tf.get_variable(
                name='fc_wnr_weights', shape=(np.prod(x.shape.as_list()[1:]), hp_fc_wnr_n_neurons),
                initializer=glorot_uniform_initializer,
            )
            fc_wnr_biases = tf.get_variable(
                name='fc_wnr_biases', shape=(hp_fc_wnr_n_neurons,), initializer=zeros_initializer,
            )
            output_wnr_weights = tf.get_variable(
                name='output_wnr_weights', shape=(hp_fc_wnr_n_neurons, 3),
                initializer=glorot_uniform_initializer,
            )
            output_wnr_biases = tf.get_variable(
                name='output_wnr_biases', shape=(3,), initializer=zeros_initializer,
            )

        with tf.name_scope('ops_output'):
            # fc_wnr = tf.keras.layers.Dense(units=hp_fc_wnr_n_neurons)(x)  #, activation='sigmoid')(x)  # todo debugging add back in activation
            # ops.output_wnr = tf.keras.layers.Dense(units=3)(fc_wnr)  # WAKE|NREM|REM classification

            # WAKE|NREM|REM classification
            fc_wnr = tf.linalg.matmul(x, fc_wnr_weights, name='fc_wnr_matmul_x_weights')  # todo try removing this layer, why does adding sigmoid cause it to fail?
            fc_wnr = tf.add(fc_wnr, fc_wnr_biases, name='fc_wnr_bias_add')
            # fc_wnr = tf.math.sigmoid(fc_wnr, name='fc_wnr_sigmoid')
            fc_wnr = tf.math.tanh(fc_wnr)

            # used for debugging, pass to comet.ml
            ops.debug_fc_mean = tf.math.reduce_mean(fc_wnr)
            ops.debug_fc_std = tf.math.reduce_std(fc_wnr)
            ops.debug_fc_wnr = fc_wnr

            output_wnr = tf.linalg.matmul(fc_wnr, output_wnr_weights, name='output_wnr_matmul_x_weights')
            ops.output_wnr = tf.add(output_wnr, output_wnr_biases, name='output_wnr_bias_add')

            fc_ma = tf.keras.layers.Dense(units=hp_fc_ma_n_neurons, activation='sigmoid')(x)
            ops.output_ma = tf.keras.layers.Dense(units=1)(fc_ma)  # MA (micro-arousal)

            fc_pa = tf.keras.layers.Dense(units=hp_fc_pa_n_neurons, activation='sigmoid')(x)
            ops.output_pa = tf.keras.layers.Dense(units=1)(fc_pa)  # PA (passive-active wake)

            fc_time = tf.keras.layers.Dense(units=hp_fc_time_n_neurons, activation='tanh')(x)
            ops.output_time = tf.keras.layers.Dense(units=1, activation='tanh')(fc_time)

    @staticmethod
    def build_optimizer(loss):
        train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss=loss, global_step=tf.train.get_or_create_global_step()
        )

        return train_op

    @staticmethod
    def build_metrics(ops, label_time, hp_max_predict_time_fps):
        # Compute metrics
        ops.accuracy_wnr, accuracy_wnr_update_op = tf.metrics.accuracy(
            labels=ops.label_wnr, predictions=ops.predicted_wnr, name='accuracy_wnr',
        )

        ops.accuracy_ma, accuracy_ma_update_op = tf.metrics.accuracy(
            labels=ops.label_ma, predictions=ops.predicted_ma, name='accuracy_ma',
        )

        ops.accuracy_pa, accuracy_pa_update_op = tf.metrics.accuracy(
            labels=ops.label_pa, predictions=ops.predicted_pa, name='accuracy_pa',
        )

        ops.rmse_time, rmse_time_update_op = tf.metrics.root_mean_squared_error(
            labels=ops.label_time, predictions=ops.predicted_time, name='metrics_rmse'
        )

        # WAKE|NREM|REM error rate for samples > 30 sec from a transition
        # hp_max_predict_time_fps determines the range for label_time in [-1, +1], assuming a frame rate of 15 fps
        long_term_state = (30 * 15) / hp_max_predict_time_fps  # 30 sec @ 15 fps over scale used to produce label_time
        is_long_term_state = tf.abs(label_time) > long_term_state
        accuracy_long_term_state, accuracy_long_term_state_update_op = tf.metrics.accuracy(
            labels=ops.label_wnr,
            predictions=ops.predicted_wnr,
            weights=tf.cast(is_long_term_state, tf.float32),
        )
        accuracy_short_term_state, accuracy_short_term_state_update_op = tf.metrics.accuracy(
            labels=ops.label_wnr,
            predictions=ops.predicted_wnr,
            weights=1.0 - tf.cast(is_long_term_state, tf.float32),
        )
        ops.error_rate_wnr_long_term_state = 1 - accuracy_long_term_state  # save error rate rather than accuracy
        ops.error_rate_wnr_short_term_state = 1 - accuracy_short_term_state  # save error rate rather than accuracy

        ops.metrics_update_op = tf.group(accuracy_wnr_update_op, accuracy_ma_update_op,
                                         accuracy_pa_update_op, rmse_time_update_op,
                                         accuracy_long_term_state_update_op, accuracy_short_term_state_update_op)

    @staticmethod
    def build_losses(ops, sleep_state, label_time_as_sampled, target_wnr, target_ma, target_pa, target_time,
                     hp_scale_loss_wnr, hp_scale_loss_ma, hp_scale_loss_pa, hp_scale_loss_time):
        # Build losses
        noop = (tf.constant(0.0, dtype=tf.float32), tf.constant([], dtype=tf.int64), tf.constant([], dtype=tf.int64))
        noop_time = (tf.constant(0.0, dtype=tf.float32), tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32))

        # todo debugging removing all other optimization objectives but WNR - this was not the issue, same effect seen
        # ops.loss_wnr, ops.label_wnr, ops.predicted_wnr = tf.cond(
        #     tf.greater(tf.count_nonzero(target_wnr), 0),
        #     true_fn=lambda: ModelSleepState.build_loss_wnr(ops, sleep_state, target_wnr),
        #     false_fn=lambda: noop,
        # )
        ops.loss_wnr, ops.label_wnr, ops.predicted_wnr = ModelSleepState.build_loss_wnr(ops, sleep_state, target_wnr)

        ops.loss_ma, ops.label_ma, ops.predicted_ma = tf.cond(
            tf.greater(tf.count_nonzero(target_ma), 0),
            true_fn=lambda: ModelSleepState.build_loss_ma(ops, sleep_state, target_ma),
            false_fn=lambda: noop,
        )

        ops.loss_pa, ops.label_pa, ops.predicted_pa = tf.cond(
            tf.greater(tf.count_nonzero(target_pa), 0),
            true_fn=lambda: ModelSleepState.build_loss_pa(ops, sleep_state, target_pa),
            false_fn=lambda: noop,
        )

        ops.loss_time, ops.label_time, ops.predicted_time = tf.cond(
            tf.greater(tf.count_nonzero(target_time), 0),
            true_fn=lambda: ModelSleepState.build_loss_time(ops, label_time_as_sampled, target_time),
            false_fn=lambda: noop_time,
        )

        # todo debugging removing all other optimization objectives but WNR - this was not the issue, same effect seen
        # ops.loss = ops.loss_wnr * hp_scale_loss_wnr + \
        #            ops.loss_ma * hp_scale_loss_ma + \
        #            ops.loss_pa * hp_scale_loss_pa + \
        #            ops.loss_time * hp_scale_loss_time
        ops.loss = ops.loss_wnr

    @staticmethod
    def build_loss_wnr(ops, sleep_state, target_wnr):
        """ WAKE|NREM|REM loss """
        # targeted_output_wnr = ops.output_wnr[target_wnr]
        # targeted_sleep_state = sleep_state[target_wnr]
        # todo debug trying without target_wnr, restore the two lines above after debugging (this was not the issue)
        targeted_output_wnr = ops.output_wnr
        targeted_sleep_state = sleep_state

        is_wake = tf.logical_or(tf.equal(targeted_sleep_state, 1), tf.equal(targeted_sleep_state, 5))
        is_nrem = tf.equal(targeted_sleep_state, 2)
        is_rem = tf.equal(targeted_sleep_state, 3)

        # loss may be nan if no samples in a particular batch target this objective
        labels_onehot_wnr = tf.cast(tf.stack((is_wake, is_nrem, is_rem), axis=1), dtype=tf.float32, name='labels_wnr')
        # todo debugging test with softmax (not the issue)
        # loss_wnr = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=labels_onehot_wnr, logits=targeted_output_wnr, name='loss_wnr'
        # )
        loss_wnr = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels_onehot_wnr, logits=targeted_output_wnr, name='loss_wnr'
        )
        loss_wnr = tf.reduce_mean(loss_wnr, name='loss_wnr')

        label_wnr = tf.argmax(labels_onehot_wnr, axis=1, name='label_wnr')
        predicted_wnr = tf.argmax(targeted_output_wnr, axis=1, name='predicted_wnr')

        # todo debugging (for debugging display)
        ops.debug_labels_onehot_wnr = labels_onehot_wnr
        ops.debug_targeted_output_wnr = targeted_output_wnr
        ops.debug_label_wnr = label_wnr
        ops.debug_predicted_wnr = predicted_wnr

        return loss_wnr, label_wnr, predicted_wnr

    @staticmethod
    def build_loss_ma(ops, sleep_state, target_ma):
        """ MA (micro-arousal) loss """
        assert sleep_state.dtype == tf.int8

        targeted_output_ma = tf.squeeze(ops.output_ma[target_ma], axis=1)
        targeted_sleep_state = sleep_state[target_ma]

        label_ma = tf.cast(tf.equal(targeted_sleep_state, 4), tf.int64)
        loss_ma = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label_ma, tf.float32), logits=targeted_output_ma, name='loss_ma'
        )
        loss_ma = tf.reduce_mean(loss_ma)
        predicted_ma = tf.cast(tf.round(targeted_output_ma), tf.int64)

        return loss_ma, label_ma, predicted_ma

    @staticmethod
    def build_loss_pa(ops, sleep_state, target_pa):
        """ PA (passive-active wake) loss """
        assert sleep_state.dtype == tf.int8

        targeted_output_pa = tf.squeeze(ops.output_pa[target_pa], axis=1)
        targeted_sleep_state = sleep_state[target_pa]

        label_pa = tf.cast(tf.equal(targeted_sleep_state, 1), tf.int64)
        loss_pa = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label_pa, tf.float32), logits=targeted_output_pa, name='loss_pa'
        )
        loss_pa = tf.reduce_mean(loss_pa)
        predicted_pa = tf.cast(tf.round(targeted_output_pa), tf.int64)

        return loss_pa, label_pa, predicted_pa

    @staticmethod
    def build_loss_time(ops, label_time_as_sampled, target_time):
        """ Time forward-backward prediction loss """
        targeted_output_time = tf.squeeze(ops.output_time[target_time], axis=1)
        targeted_label_time = label_time_as_sampled[target_time]

        loss_time = tf.losses.mean_squared_error(
            labels=targeted_label_time, predictions=targeted_output_time,
        )
        loss_time = tf.reduce_mean(loss_time)

        return loss_time, targeted_label_time, targeted_output_time

    @staticmethod
    def build_cnn(input_neural, name_scope='ops_timequantcnn', variable_scope='vars_timequantcnn') -> tf.Tensor:
        # Expected input shape: [batch, segment_length, channels]

        msra_initializer = tf.initializers.variance_scaling(scale=2.0, mode='fan_in', distribution='normal')

        batch_size, segment_length, n_channels = input_neural.shape

        # Create filters
        # with tf.variable_scope(variable_scope):
        #     filters0 = tf.get_variable(
        #         name='conv0_filters', shape=(min(30, math.ceil(segment_length / 4**1)), n_channels, 320),
        #         initializer=msra_initializer
        #     )
        #     filters1 = tf.get_variable(
        #         name='conv1_filters', shape=(min(30, math.ceil(segment_length / 4**2)), 320, 384),
        #         initializer=msra_initializer
        #     )
        #     filters2 = tf.get_variable(
        #         name='conv2_filters', shape=(min(30, math.ceil(segment_length / 4**3)), 384, 448),
        #         initializer=msra_initializer
        #     )
        #     filters3 = tf.get_variable(
        #         name='conv3_filters', shape=(min(30, math.ceil(segment_length / 4**4)), 448, 512),
        #         initializer=msra_initializer
        #     )
        #     filters4 = tf.get_variable(
        #         name='conv4_filters', shape=(min(30, math.ceil(segment_length / 4**5)), 512, 512),
        #         initializer=msra_initializer
        #     )
        #     filters5 = tf.get_variable(
        #         name='conv5_filters', shape=(min(30, math.ceil(segment_length / 4**6)), 512, 512),
        #         initializer=msra_initializer
        #     )
        #     filters6 = tf.get_variable(
        #         name='conv6_filters', shape=(min(30, math.ceil(segment_length / 4**7)), 512, 512),
        #         initializer=msra_initializer
        #     )
        #     filters7 = tf.get_variable(
        #         name='conv7_filters', shape=(min(30, math.ceil(segment_length / 4**8)), 512, 512),
        #         initializer=msra_initializer
        #     )
        #     filters8 = tf.get_variable(
        #         name='conv8_filters', shape=(min(30, math.ceil(segment_length / 4**9)), 512, 512),
        #         initializer=msra_initializer
        #     )
        #     filters9 = tf.get_variable(
        #         name='conv9_filters', shape=(min(30, math.ceil(segment_length / 4**10)), 512, 512),
        #         initializer=msra_initializer
        #     )

        # # Create conv operations
        # with tf.name_scope(name_scope):
        #     x = input_neural
        #     x = tf.nn.conv1d(input=x, filters=filters0, stride=4, padding='SAME', name='conv0_layer')
        #     x = tf.nn.relu(x, name='conv0_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters1, stride=4, padding='SAME', name='conv1_layer')
        #     x = tf.nn.relu(x, name='conv1_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters2, stride=4, padding='SAME', name='conv2_layer')
        #     x = tf.nn.relu(x, name='conv2_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters3, stride=4, padding='SAME', name='conv3_layer')
        #     x = tf.nn.relu(x, name='conv3_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters4, stride=4, padding='SAME', name='conv4_layer')
        #     x = tf.nn.relu(x, name='conv4_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters5, stride=4, padding='SAME', name='conv5_layer')
        #     x = tf.nn.relu(x, name='conv5_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters6, stride=4, padding='SAME', name='conv6_layer')
        #     x = tf.nn.relu(x, name='conv6_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters7, stride=4, padding='SAME', name='conv7_layer')
        #     x = tf.nn.relu(x, name='conv7_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters8, stride=4, padding='SAME', name='conv8_layer')
        #     x = tf.nn.relu(x, name='conv8_relu')
        #     x = tf.nn.conv1d(input=x, filters=filters9, stride=4, padding='SAME', name='conv9_layer')
        #     x = tf.nn.relu(x, name='conv9_relu')

        # todo v702 using original convolutional network parameters only supporting 65536
        with tf.variable_scope(variable_scope):
            filters0 = tf.get_variable(name='conv0_filters', shape=(30, n_channels, 320), initializer=msra_initializer)
            filters1 = tf.get_variable(name='conv1_filters', shape=(30, 320, 384), initializer=msra_initializer)
            filters2 = tf.get_variable(name='conv2_filters', shape=(30, 384, 448), initializer=msra_initializer)
            filters3 = tf.get_variable(name='conv3_filters', shape=(30, 448, 512), initializer=msra_initializer)
            filters4 = tf.get_variable(name='conv4_filters', shape=(30, 512, 512), initializer=msra_initializer)
            filters5 = tf.get_variable(name='conv5_filters', shape=(30, 512, 512), initializer=msra_initializer)
            filters6 = tf.get_variable(name='conv6_filters', shape=(30, 512, 512), initializer=msra_initializer)
            filters7 = tf.get_variable(name='conv7_filters', shape=(30, 512, 512), initializer=msra_initializer)

        with tf.name_scope(name_scope):
            x = input_neural
            x = tf.nn.conv1d(input=x, filters=filters0, stride=4, padding='SAME', name='conv0_layer')
            x = tf.nn.relu(x, name='conv0_relu')
            x = tf.nn.conv1d(input=x, filters=filters1, stride=4, padding='SAME', name='conv1_layer')
            x = tf.nn.relu(x, name='conv1_relu')
            x = tf.nn.conv1d(input=x, filters=filters2, stride=4, padding='SAME', name='conv2_layer')
            x = tf.nn.relu(x, name='conv2_relu')
            x = tf.nn.conv1d(input=x, filters=filters3, stride=4, padding='SAME', name='conv3_layer')
            x = tf.nn.relu(x, name='conv3_relu')
            x = tf.nn.conv1d(input=x, filters=filters4, stride=4, padding='SAME', name='conv4_layer')
            x = tf.nn.relu(x, name='conv4_relu')
            x = tf.nn.conv1d(input=x, filters=filters5, stride=4, padding='SAME', name='conv5_layer')
            x = tf.nn.relu(x, name='conv5_relu')
            x = tf.nn.conv1d(input=x, filters=filters6, stride=4, padding='SAME', name='conv6_layer')
            x = tf.nn.relu(x, name='conv6_relu')
            x = tf.nn.conv1d(input=x, filters=filters7, stride=4, padding='SAME', name='conv7_layer')
            # x = tf.nn.relu(x, name='conv7_relu')

        return x


def inference_on_dataset(labels_matrix: np.ndarray,
                         video_files: (tuple, list),
                         neural_files: (tuple, list),
                         neural_files_basepath: str,
                         sample_width: int,
                         n_channels: int,
                         load_model: str = None,
                         corrupt_neural_files: (tuple, list, str) = None,
                         include_video_files: (tuple, list) = None,
                         local_cache_size: int = 50e+9,
                         local_cache_workers: int = 4,
                         include_channels: str = None,
                         batch_size: int = 20,
                         callback_output_processor: typing.Callable = None,
                         hp_max_predict_time_fps: int = None,
                         ) -> dict:
    """
    Performs inference on a set of results in labels_matrix

    :param local_cache_workers:
    :param labels_matrix:
    :param video_files:
    :param neural_files:
    :param neural_files_basepath:
    :param sample_width:
    :param n_channels:
    :param load_model:
    :param corrupt_neural_files:
    :param include_video_files: optionally limit predictions to the specified subset of video file names or ixs.
    :param local_cache_size:
    :param include_channels:
    :param batch_size:
    :param callback_output_processor: if provided each batch will be passed to this function for streaming
        processing instead of accumulating it.
    :param hp_max_predict_time_fps:
    :return: a dictionary containing each output value as a numpy array of the length of labels_matrix
    """
    assert hp_max_predict_time_fps is not None, \
        'This value is required, it will cause errors if it is omitted even if unused.'

    include_channels_list = parse_channels_list(include_channels)
    n_channels = parse_n_channels(n_channels, neural_files)
    corrupt_neural_files = corrupt_neural_files.split(':') \
        if isinstance(corrupt_neural_files, str) else corrupt_neural_files
    n_devices = max(1, len(tf.config.list_logical_devices('GPU')))

    generate_sleep_state = dataset.GenerateSleepState(
        labels_file=(labels_matrix, video_files, neural_files), sample_width=sample_width, n_channels=n_channels,
        samples_per_partition=50, neural_files_basepath=neural_files_basepath, sequential_batch_size=batch_size,
        include_channels=include_channels_list, include_video_files=include_video_files,
        exclude_neural_files=corrupt_neural_files, hp_max_predict_time_fps=hp_max_predict_time_fps,
    )
    dataset_growing_epochs = growing_epochs.DatasetGrowingEpochs(
        sample_generator=generate_sleep_state, local_cache_size=local_cache_size,
        n_workers=local_cache_workers, sequential=True
    )
    ds = dataset_growing_epochs.as_dataset()
    ds = ds.prefetch(buffer_size=n_devices)
    get_next_tensor = tf.data.make_one_shot_iterator(ds).get_next()

    # dataset (1 per gpu) prefetch to GPU
    devices = tf.config.list_logical_devices('GPU') if len(tf.config.list_logical_devices('GPU')) > 0 \
        else [SimpleNamespace(name='CPU:0')]
    print('Using devices: ', [d.name for d in devices])

    result_processor = queue.Queue() if callback_output_processor is None else callback_output_processor
    progress = SimpleNamespace(lock=threading.Lock(), count=0, last_update=time.time(),
                               sample_count=generate_sleep_state.labels_matrix.shape[0])

    sess = model_base.create_session()

    # Compute predictions using multiple threads/GPUs using a producer/consumer pattern
    predict_consumer_threads = []
    with sess:
        for device in devices:
            scope = re.sub('/|device|:', '', device.name)
            print('Building model on device: {} with scope {}'.format(device.name, scope))
            model = ModelSleepState(session=sess, device=device.name, checkpoint_period_min=60, scope=scope)
            model.build_model(input_tensors=get_next_tensor,
                              include_channels=include_channels_list, n_channels=n_channels,
                              sample_width=sample_width, build_loss=False,
                              hp_max_predict_time_fps=hp_max_predict_time_fps)
            if load_model is not None:
                model.load_model(load_model)
            else:
                model.init()
            thread = threading.Thread(
                target=_predict_consumer_thread,
                args=(model, get_next_tensor, result_processor, progress),
                name='predict_consumer_thread_{}'.format(scope),
            )
            thread.start()
            predict_consumer_threads.append(thread)
        for t in predict_consumer_threads:
            t.join()

    if callback_output_processor is None:
        output_accumulator = list(result_processor.queue)
        merged_output = {k: np.concatenate([o[k] for o in output_accumulator]) for k in output_accumulator[0].keys()}
        return merged_output


def _predict_consumer_thread(model, get_next_tensor, result_processor: (queue.Queue, typing.Callable), progress):
    """ Runs the model in a loop consuming elements from get_next_tensor and returns the results to the result_queue """
    fetch_ops = {  # filtered_get_next_tensor
        k: v for k, v in get_next_tensor.items() if k not in ['neural_data', 'neural_length']
    }
    fetch_ops['output_wnr'] = model.ops.output_wnr
    fetch_ops['output_ma'] = model.ops.output_ma
    fetch_ops['output_pa'] = model.ops.output_pa
    fetch_ops['output_time'] = model.ops.output_time

    try:
        while True:
            output = model.session.run(fetches=fetch_ops)

            if callable(result_processor):
                callback_output_processor(output)
            elif isinstance(result_processor, queue.Queue):
                result_processor.put(output)
            else:
                raise ValueError('result_processor must be a queue or a function')

            with progress.lock:
                progress.count += output['sleep_states'].shape[0]
                t = time.time()
                if t - progress.last_update > 15:
                    print('Predict progress: {}/{}  {:0.0f}s  thread: {}'.format(
                        progress.count, progress.sample_count, t - progress.last_update, threading.current_thread().name
                    ))
                    progress.last_update = t

    except tf.errors.OutOfRangeError:
        with progress.lock:
            print('Predict progress: {}/{}  {:0.0f}s  exiting thread: {}'.format(
                progress.count, progress.sample_count, t - progress.last_update, threading.current_thread().name
            ))


def summary_statistics(label_sleepstate, label_time,
                       predicted_wnr, predicted_ma, predicted_pa,
                       predicted_time):
    """ Generates summary statistics for a set of predictions """

    mask_wake = np.isin(label_sleepstate, [1, 5])
    mask_nrem = label_sleepstate == 2
    mask_rem = label_sleepstate == 3

    mask_time = np.abs(label_time) <= 1.0

    acc_wake = np.sum(predicted_wnr[mask_wake] == 0) / np.sum(mask_wake)
    acc_nrem = np.sum(predicted_wnr[mask_nrem] == 1) / np.sum(mask_nrem)
    acc_rem = np.sum(predicted_wnr[mask_rem] == 2) / np.sum(mask_rem)
    acc_pa = np.sum((((predicted_pa[mask_wake] - 1) * -4) + 1) == label_sleepstate[mask_wake]) / np.sum(mask_wake) \
        if all(predicted_pa >= 0) else math.nan
    f1_ma = sklearn.metrics.f1_score(
        y_true=(label_sleepstate == 4).astype(predicted_ma.dtype),
        y_pred=predicted_ma
    ) if len(predicted_ma) > 0 and all(predicted_ma >= 0) else math.nan
    rmse_time = np.sqrt(((label_time[mask_time] - predicted_time[mask_time])**2).mean())

    n_wnr = np.sum(mask_wake ^ mask_nrem ^ mask_rem)
    acc_wnr = acc_wake * (np.sum(mask_wake) / n_wnr) + \
                acc_nrem * (np.sum(mask_nrem) / n_wnr) + \
                acc_rem * (np.sum(mask_rem) / n_wnr)

    return acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time


def read_file_bytes(filename):
    """ Reads a file bytes, the file may be local or S3 and may be zipped (single file) or not. Bytes returned."""
    print('Getting file: {}'.format(filename))
    bytes_raw = smart_open.open(filename, 'rb').read()
    if filename.endswith('.zip'):
        z = zipfile.ZipFile(io.BytesIO(bytes_raw))
        bytes_raw = z.read(z.infolist()[0])
    return bytes_raw


def generate_summary_statistics(predictions_filename: (str, io.IOBase), output_filename: (str, io.IOBase),
                                test_video_files: str = None, display: bool = True):
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
    test_video_files = () if test_video_files is None else test_video_files.split(',')
    predictions_bytes = read_file_bytes(predictions_filename)
    predictions = np.recfromtxt(io.BytesIO(predictions_bytes), names=True, delimiter=',')

    # test-set
    output += ['per-label-test-set']
    predictions_filtered = predictions[np.isin(predictions['video_filename'].astype(np.str), test_video_files)]
    acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time = summary_statistics(
        label_sleepstate=predictions_filtered['label'],
        label_time=predictions_filtered['label_time'],
        predicted_wnr=predictions_filtered['predicted_wnr_012'],
        predicted_ma=predictions_filtered['predicted_ma_01'],
        predicted_pa=predictions_filtered['predicted_pa_01'],
        predicted_time=predictions_filtered['predicted_time'],
    )
    output += _formatter_dataset(acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma)
    output += ['']

    # train-set
    output += ['per-label-train-set']
    predictions_filtered = predictions[~np.isin(predictions['video_filename'].astype(np.str), test_video_files)]
    acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time = summary_statistics(
        label_sleepstate=predictions_filtered['label'],
        label_time=predictions_filtered['label_time'],
        predicted_wnr=predictions_filtered['predicted_wnr_012'],
        predicted_ma=predictions_filtered['predicted_ma_01'],
        predicted_pa=predictions_filtered['predicted_pa_01'],
        predicted_time=predictions_filtered['predicted_time'],
    )
    output += _formatter_dataset(acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma)
    output += ['']

    # per-video-filename-accuracy
    output += ['per-video-metrics']
    output += ['                                   Passive/   Micro-']
    output += ['    WAKE   |   NREM   |   REM    |  Active  |  arousal | video-filename']
    output += ['  -------- | -------- | -------- | -------- | -------- | ---------------']
    for video in np.unique(predictions['video_filename']):
        video = video.decode()
        is_test_video = True if np.isin(video, test_video_files) else False
        predictions_filtered = predictions[predictions['video_filename'].astype(np.str) == video]

        if len(predictions_filtered) > 0:
            acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time = summary_statistics(
                label_sleepstate=predictions_filtered['label'],
                label_time=predictions_filtered['label_time'],
                predicted_wnr=predictions_filtered['predicted_wnr_012'],
                predicted_ma=predictions_filtered['predicted_ma_01'],
                predicted_pa=predictions_filtered['predicted_pa_01'],
                predicted_time=predictions_filtered['predicted_time'],
            )
        else:
            acc_wake = acc_nrem = acc_rem = acc_pa = f1_ma = math.nan

        output += ['    {:.2f}   |   {:.2f}   |   {:.2f}   |   {:.2f}   |   {:.2f}   | {}{}'
                       .format(acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, video,
                               ' (test-set)' if is_test_video else '').replace('nan', ' NaN')]

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


def _formatter_dataset(acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma):
    output = []
    output += ['   {:.2f}: WAKE            (accuracy)'.format(acc_wake).replace('nan', ' NaN')]
    output += ['   {:.2f}: NREM            (accuracy)'.format(acc_nrem).replace('nan', ' NaN')]
    output += ['   {:.2f}: REM             (accuracy)'.format(acc_rem).replace('nan', ' NaN')]
    output += ['   {:.2f}: Passive/Active  (accuracy)'.format(acc_pa).replace('nan', ' NaN')]
    output += ['   {:.2f}: Micro-arousal   (F1 score)'.format(f1_ma).replace('nan', ' NaN')]
    return output

    
def parse_channels_list(channels: str):
    """ Parses a comma separated string into a list of ints. """
    if channels is None or channels == '':
        result = None
    else:
        # split on comma
        result = channels.split(',')
        # resolve ranges
        result = [x.split(':') for x in result]
        result = [(list(range(int(x[0]), int(x[1]))) if len(x) == 2 else [int(x[0])]) for x in result]
        result = list(itertools.chain.from_iterable(result))

    return result


def parse_n_channels(n_channels: int, labels_file_or_neural_files: (str, list, tuple)):
    """
    Returns n_channels if it's an int already, or parses n_channels
    from filenames in labels_file, or from neural_files list directly
    """
    if n_channels is None:
        if isinstance(labels_file_or_neural_files, str):
            _, _, neural_files = dataset.load_labels_file(labels_file_or_neural_files)
        else:
            neural_files = labels_file_or_neural_files

        n_channels = int(re.findall(r'_(\d*)_Channels', neural_files[0])[0])
    return n_channels


def train(training_steps: int, labels_file: str, hp_max_predict_time_fps: int,
          checkpoint: str = None, checkpoint_final: str = None,
          load_model: str = None, disable_comet: bool = False, neural_files_basepath: str = None, n_channels: int = None,
          include_channels: str = None, sample_width: int = None, corrupt_neural_files: (tuple, list, str) = None,
          test_video_files: (tuple, list, str) = None, local_cache_size: int = 50e+9, local_cache_workers: int = 4,
          batch_size: int = 8, include_modules: str = None,
          testeval_on_checkpoint: bool = True, data_echo_factor:int = 2, n_workers:int = 16,
          ):

    include_channels_list = parse_channels_list(include_channels)
    n_channels = parse_n_channels(n_channels, labels_file)
    corrupt_neural_files = corrupt_neural_files.split(':') if isinstance(corrupt_neural_files, str) else corrupt_neural_files
    test_video_files = [test_video_files] if isinstance(test_video_files, str) else test_video_files

    # Sanity check that we don't leave debug settings accidentally on
    if training_steps < 1000:
        print('*' * 200 + '\n' * 5 + 'WARNING - TRAINING STEPS BELOW 1000' + '\n' * 5 + '*' * 200)

    print('Excluding test video files {}, and corrupt neural files {}'.format(test_video_files, corrupt_neural_files))

    # For comet output logging and progress bar issues try: 'simple', 'native', or None
    experiment = comet_ml.Experiment(api_key='uCZTtx2dLDpXYFu0aXzmgygn6', project_name='sleep-state-model',
                                     disabled=disable_comet, auto_output_logging='simple')
    experiment.set_name(os.getenv('HOSTNAME')[16:])

    # Model
    model = ModelSleepState()

    # # todo debugging
    # generate_sleep_state = dataset.GenerateSleepStateTrainBatch(
    #     labels_file=labels_file, sample_width=sample_width, n_channels=n_channels,
    #     hp_max_predict_time_fps=hp_max_predict_time_fps,
    #     neural_files_basepath=neural_files_basepath, include_channels=include_channels_list,
    #     exclude_video_files=test_video_files, exclude_neural_files=corrupt_neural_files,
    #     include_modules=include_modules,
    #     partition_size=int(1e+9) if training_steps > 1000 else 1,
    # )
    # dataset_growing_epochs = growing_epochs.DatasetGrowingEpochs(
    #     sample_generator=generate_sleep_state, local_cache_size=100000000000, n_workers=4,
    #     sequential=False, n_parallel_reads=4
    # )
    # ds = dataset_growing_epochs.as_dataset()
    dataset_sleepstate_train = dataset.DatasetSleepStateTrain(
        labels_file=labels_file, sample_width=sample_width, n_channels=n_channels,
        hp_max_predict_time_fps=hp_max_predict_time_fps, include_modules=include_modules,
        neural_files_basepath=neural_files_basepath, include_channels=include_channels_list,
        exclude_video_files=test_video_files, exclude_neural_files=corrupt_neural_files,
        n_workers=n_workers,
    )

    ds = dataset_sleepstate_train.as_dataset()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    ds = ds.flat_map(lambda t: tf.data.Dataset.from_tensors(t).repeat(data_echo_factor))
    ds = ds.apply(tf.data.experimental.prefetch_to_device(
        device='gpu:0' if tf.test.is_gpu_available(cuda_only=True) else 'cpu', buffer_size=1
    ))

    get_next_tensor = tf.data.make_one_shot_iterator(ds).get_next()

    model.build_model(
        input_tensors=get_next_tensor,
        sample_width=sample_width,
        n_channels=n_channels,
        include_channels=include_channels_list,
        hp_max_predict_time_fps=hp_max_predict_time_fps
    )
    experiment.set_model_graph(tf.get_default_graph())

    model.init()

    if load_model:
        model.load_model(load_model)

    # Training loop
    fetch_ops = {
        # Train ops
        'train_op': model.ops.train_op,
        'metrics_update_op': model.ops.metrics_update_op,

        # Loss
        'loss': model.ops.loss,

        # Targets
        'target_wnr': get_next_tensor['target_wnr'],
        'target_ma': get_next_tensor['target_ma'],
        'target_pa': get_next_tensor['target_pa'],
        'target_time': get_next_tensor['target_time'],

        # Debug  todo remove after debugging complete
        # 'sample_ix': get_next_tensor['sample_ix'],
        'output_wnr': model.ops.output_wnr,
        'sleep_states': get_next_tensor['sleep_states'],
        # 'debug_fc_wnr': model.ops.debug_fc_wnr,
        'debug_labels_onehot_wnr': model.ops.debug_labels_onehot_wnr,
        'debug_targeted_output_wnr': model.ops.debug_targeted_output_wnr,
        'debug_label_wnr': model.ops.debug_label_wnr,
        'debug_predicted_wnr': model.ops.debug_predicted_wnr,
    }

    # Set initial global step (will be non-zero if an existing model was loaded)
    out = SimpleNamespace(**model.session.run(fetches={'global_step': model.ops.global_step}))
    last_step_time = time.time()  # used to compute the time per training step for printing to stdout

    metrics_update_frequency = 200
    loss_update_frequency = 10

    # Training loop
    while out.global_step < training_steps:

        # Select optional fetch_ops based on next global step
        fetch_ops_metrics = {}
        if (out.global_step + 1) % metrics_update_frequency == 0 or out.global_step == 0:
            # Metrics
            if 'wnr' in include_modules:
                fetch_ops_metrics['loss_wnr'] = model.ops.loss_wnr
                fetch_ops_metrics['accuracy_wnr'] = model.ops.accuracy_wnr
                fetch_ops_metrics['error_rate_wnr_long_term_state'] = model.ops.error_rate_wnr_long_term_state
                fetch_ops_metrics['error_rate_wnr_short_term_state'] = model.ops.error_rate_wnr_short_term_state
                fetch_ops_metrics['debug_fc_mean'] = model.ops.debug_fc_mean  # debugging
                fetch_ops_metrics['debug_fc_std'] = model.ops.debug_fc_std  # debugging
            if 'ma' in include_modules:
                fetch_ops_metrics['loss_ma'] = model.ops.loss_ma
                fetch_ops_metrics['accuracy_ma'] = model.ops.accuracy_ma
            if 'pa' in include_modules:
                fetch_ops_metrics['loss_pa'] = model.ops.loss_pa
                fetch_ops_metrics['accuracy_pa'] = model.ops.accuracy_pa
            if 'time' in include_modules:
                fetch_ops_metrics['loss_time'] = model.ops.loss_time
                fetch_ops_metrics['rmse_time'] = model.ops.rmse_time

        other_ops = {}
        if (out.global_step + 1) % metrics_update_frequency == 0 or out.global_step == 0 and 'time' in include_modules:
            # Sanity check ops for time, output to html in comet.ml
            other_ops['label_time'] = get_next_tensor['label_time']
            other_ops['label_time_as_sampled'] = get_next_tensor['label_time_as_sampled']
            other_ops['output_time'] = model.ops.output_time

        # Executes sess.run, fetches everything in fetch_ops, and returns results to out which
        # supports dot-notation access to each output by name
        out = SimpleNamespace(**model.session.run({**fetch_ops, **fetch_ops_metrics, **other_ops}),
                              global_step=model.session.run(model.ops.global_step))

        # Comet.ml metrics
        if out.global_step % loss_update_frequency == 0:
            experiment.log_metric('loss', out.loss, step=out.global_step)
        for k, v in fetch_ops_metrics.items():
            experiment.log_metric(k, out[k], step=out.global_step)

        # Reset local variables for metrics
        if any(k in fetch_ops_metrics for k in ['accuracy_wnr', 'accuracy_ma', 'accuracy_pa', 'rmse_time']):
            model.session.run(tf.local_variables_initializer())

        # Other reporting
        if 'label_time' in vars(out):
            # table, th, td {border: 1px solid black;} th, td {padding: 4px;}
            html = '<style>table, th, td {border: 1px solid black;} th, td {padding: 4px;}</style>'
            html += '<table>'
            html += '<tr><th>Step</th><th>{}</th><th></th></tr>'.format(out.global_step)
            html += '<tr><th>label_time</th><th>label_time_as_sampled</th><th>output_time</th></tr>'
            for label_time, label_time_as_sampled, output_time in zip(out.label_time, out.label_time_as_sampled, out.output_time):
                html += '<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>'.format(
                    label_time, label_time_as_sampled, output_time[0]
                )
            html += '</table>'
            experiment.log_html(html=html)

        # Example: 'metrics (wnr) (ma) (pa) (time) (0.012) (0.128) (0.963) (0.542)
        metrics_string = 'metrics: {wnr}{ma}{pa}{time} {accuracy_wnr}{error_rate_wnr}{accuracy_ma}{accuracy_pa}{rmse_time}'.format(
            wnr='(wnr) ' if 'accuracy_wnr' in fetch_ops_metrics else '',
            ma='(ma) ' if 'accuracy_ma' in fetch_ops_metrics else '',
            pa='(pa) ' if 'accuracy_pa' in fetch_ops_metrics else '',
            time='(time) ' if 'rmse_time' in fetch_ops_metrics else '',
            accuracy_wnr='({:.3f}) '.format(out.accuracy_wnr) if 'accuracy_wnr' in fetch_ops_metrics else '',
            accuracy_ma='({:.3f}) '.format(out.accuracy_ma) if 'accuracy_ma' in fetch_ops_metrics else '',
            accuracy_pa='({:.3f}) '.format(out.accuracy_pa) if 'accuracy_pa' in fetch_ops_metrics else '',
            rmse_time='({:.3f}) '.format(out.rmse_time) if 'rmse_time' in fetch_ops_metrics else '',
            error_rate_wnr=' ({:.3f}/{:.3f}) '.format(
                out.error_rate_wnr_long_term_state, out.error_rate_wnr_short_term_state
            ) if 'error_rate_wnr_long_term_state' in fetch_ops_metrics else '',
        )

        loss_string = 'loss: {loss:.5f}{loss_wnr}{loss_ma}{loss_pa}{loss_time}'.format(
            loss=out.loss,
            loss_wnr=' wnr {:.5f}'.format(out.loss_wnr) if 'loss_wnr' in fetch_ops_metrics else '',
            loss_ma=' ma {:.5f}'.format(out.loss_ma) if 'loss_ma' in fetch_ops_metrics else '',
            loss_pa=' pa {:.5f}'.format(out.loss_pa) if 'loss_pa' in fetch_ops_metrics else '',
            loss_time=' time {:.5f}'.format(out.loss_time) if 'rmse_time' in fetch_ops_metrics else '',
        )

        # debugging
        if hasattr(out, 'debug_labels_onehot_wnr'):
            print('DEBUG> \n\n{}\n{}\n{}\n{}'.format(out.debug_label_wnr, out.debug_predicted_wnr, out.debug_labels_onehot_wnr, out.debug_targeted_output_wnr))

        # Print progress
        step_time = time.time() - last_step_time
        last_step_time = last_step_time + step_time
        print('{global_step:d}/{training_steps:d} | {step_time:.2f}s | {loss_string} | {metrics_string}'.format(
            global_step=out.global_step,
            training_steps=training_steps,
            step_time=step_time,
            loss_string=loss_string,
            metrics_string=metrics_string,
        ))

        # Print debug
        if 'labels_time' in fetch_ops_metrics:
            print('DEBUG> labels_time:\n{}\noutput_time_slices\n{}\n-------------------------'.format(
                out.labels_time, model_base.ModelBase.sigmoid(out.output_time_slices)
            ))

        chkpt_file = model.save_checkpoint_periodically(checkpoint, out.global_step, out.loss, force=False)
        if chkpt_file is not None and testeval_on_checkpoint:
            submit_test_eval_job(chkpt_file, ','.join(test_video_files), experiment.get_key(), include_modules)

    # Save final checkpoint
    chkpt_file = model.save_checkpoint_periodically(
        checkpoint_final if checkpoint_final is not None else checkpoint, out.global_step, out.loss, force=True
    )
    if testeval_on_checkpoint:
        submit_test_eval_job(chkpt_file, ','.join(test_video_files), experiment.get_key(), include_modules)

    # dataset_growing_epochs._shutdown()
    print('Exiting train loop')
    experiment.end()
    return model, out.loss  # these outputs are for unittests or api usage


def submit_test_eval_job(chkpt_file: str, include_video_files: str, experiment_key: str, include_modules: str):
    """
    Submits a new kubernetes job that performs testset evaluation on an intermediate checkpoint.
    :param chkpt_file: Checkpoint path on S3
    :param experiment_key: Comet.ml experiment key to submit results to, Experiment.get_key()
    :param include_video_files: Test set video files as a CSV string
    :param include_modules: CSV string of modules to include and report on
    """
    os.environ['MODEL'] = chkpt_file
    os.environ['EXPERIMENT_KEY'] = experiment_key
    os.environ['INCLUDE_VIDEO_FILES'] = include_video_files
    os.environ['INCLUDE_MODULES'] = include_modules
    os.system('envsubst < ../k8s/job_model_sleepstate_testeval.yaml | kubectl create -f -')


def write_result(output_file: str, output: dict, video_files: np.ndarray, neural_files: np.ndarray):
    ss_text = {-1: 'NA', 0: 'Error', 1: 'Wake_Active', 2: 'NREM', 3: 'REM', 4: 'Wake_Passive', 5: 'Micro_Arousal'}
    na = np.zeros_like(output['sleep_states']) - 1

    probability_wnr = model_base.ModelBase.softmax(output['output_wnr'])
    probability_ma = np.squeeze(model_base.ModelBase.sigmoid(output['output_ma']))
    probability_pa = np.squeeze(model_base.ModelBase.sigmoid(output['output_pa']))
    output_time = np.squeeze(output['output_time'])
    predicted_wnr = np.argmax(probability_wnr, axis=-1)
    predicted_pa = np.squeeze(np.round(probability_pa).astype(np.int32)) if 'output_pa' in output else na
    predicted_ma = np.squeeze(np.round(probability_ma).astype(np.int32)) if 'output_ma' in output else na

    results = [
        '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
            output['sleep_states'][i],           # label
            ss_text[output['sleep_states'][i]],  # label text
            predicted_wnr[i],                    # predicted value for WAKE|NREM|REM
            predicted_ma[i],                     # predicted value for MICRO-AROUSAL
            predicted_pa[i],                     # predicted value for PASSIVE|ACTIVE
            probability_wnr[i][0],               # softmax value for WAKE
            probability_wnr[i][1],               # softmax value for NREM
            probability_wnr[i][2],               # softmax value for REM
            probability_ma[i],                   # probability for MICRO-AROUSAL [0==False, 1==True]
            probability_pa[i],                   # probability for PASSIVE(0) | ACTIVE(1)
            output['next_wake_state'][i],        # label next_wake_state
            output['last_wake_state'][i],        # label last_wake_state
            output['next_nrem_state'][i],        # label next_nrem_state
            output['last_nrem_state'][i],        # label last_nrem_state
            output['next_rem_state'][i],         # label next_rem_state
            output['last_rem_state'][i],         # label last_rem_state
            output['label_time'][i],             # calculated time prediction label
            output_time[i],                      # time prediction in [-1, 1]
            os.path.basename(video_files[output['video_filename_ixs'][i]]),    # video filename
            output['video_frame_offsets'][i],    # video frame number
            os.path.basename(neural_files[output['neural_filename_ixs'][i]]),  # neural filename
            output['neural_offsets'][i],         # neural offset
        )
        for i in range(output['output_wnr'].shape[0])
    ]
    if output_file is not None:
        with smart_open.open(output_file, 'a+') as f:
            for result in results:
                f.writelines(result + '\n')
    else:
        for result in results:
            print(result)


def testeval(experiment_key: str, include_modules: str,
             load_model: str, labels_file: str, include_channels: str = None,
             neural_files_basepath: str = None, sample_width: int = None, n_channels: int = None,
             corrupt_neural_files: (tuple, list, str) = None, include_video_files: (tuple, list) = None,
             limit_samples: int = None, local_cache_size: int = 50e+9, local_cache_workers: int = 4,
             hp_max_predict_time_fps: int = None):

    output_file = '/tmp/predictions.csv'
    global_step = tf.train.load_variable(load_model, 'global_step')

    # Generate prediction CSV
    predict(load_model, labels_file, output_file, include_channels,
            neural_files_basepath, sample_width, n_channels,
            corrupt_neural_files, include_video_files,
            limit_samples, local_cache_size, local_cache_workers, hp_max_predict_time_fps)

    # Run summary statistics
    predictions_bytes = read_file_bytes(output_file + '.zip')
    predictions = np.recfromtxt(io.BytesIO(predictions_bytes), names=True, delimiter=',')

    # test-set
    predictions_filtered = predictions[np.isin(predictions['video_filename'].astype(np.str), include_video_files)]
    acc_wnr, acc_wake, acc_nrem, acc_rem, acc_pa, f1_ma, rmse_time = summary_statistics(
        label_sleepstate=predictions_filtered['label'],
        label_time=predictions_filtered['label_time'],
        predicted_wnr=predictions_filtered['predicted_wnr_012'],
        predicted_ma=predictions_filtered['predicted_ma_01'],
        predicted_pa=predictions_filtered['predicted_pa_01'],
        predicted_time=predictions_filtered['predicted_time'],
    )

    # Report result to comet
    experiment = comet_ml.ExistingExperiment(api_key='uCZTtx2dLDpXYFu0aXzmgygn6', previous_experiment=experiment_key)
    if 'wnr' in include_modules:
        experiment.log_metric('test_accuracy_wnr', acc_wnr, step=global_step)
        print('Test accuracy wnr: {} at step {}'.format(acc_wnr, global_step))
    if 'ma' in include_modules:
        experiment.log_metric('test_f1score_ma', f1_ma, step=global_step)
        print('Test f1 score ma: {} at step {}'.format(f1_ma, global_step))
    if 'pa' in include_modules:
        experiment.log_metric('test_accuracy_pa', acc_pa, step=global_step)
        print('Test accuracy pa: {} at step {}'.format(acc_pa, global_step))
    if 'time' in include_modules:
        experiment.log_metric('test_rmse_time', rmse_time, step=global_step)
        print('Test rmse time: {} at step {}'.format(rmse_time, global_step))


# noinspection PyUnusedLocal
def predict(load_model: str, labels_file: str, output_file: str, include_channels: str = None,
            neural_files_basepath: str = None, sample_width: int = None, n_channels: int = None,
            corrupt_neural_files: (tuple, list, str) = None, include_video_files: (tuple, list) = None,
            limit_samples: int = None, local_cache_size: int = 50e+9, local_cache_workers: int = 4,
            hp_max_predict_time_fps: int = None, include_modules: str = None):
    """ Generates predictions per each frame in the labels file """
    n_channels = parse_n_channels(n_channels, labels_file)

    assert output_file is None or not os.path.exists(output_file), \
        'Output file {} exists already.'.format(output_file)
    output_file_tmp = '/tmp/{}'.format(uuid.uuid1())

    # Write CSV header to output_file
    csv_header = 'label,label_text,' \
                 'predicted_wnr_012,' \
                 'predicted_ma_01,' \
                 'predicted_pa_01,' \
                 'probability_wake,probability_nrem,probability_rem,' \
                 'probability_micro_arousal,' \
                 'probability_passive_active,' \
                 'label_next_wake,label_last_wake,' \
                 'label_next_nrem,label_last_nrem,' \
                 'label_next_rem,label_last_rem,' \
                 'label_time,predicted_time,' \
                 'video_filename,video_frame_ix,neural_filename,neural_offset\n'
    if output_file_tmp is not None:
        with smart_open.open(output_file_tmp, 'a+') as f:
            f.writelines(csv_header)
    else:
        print(csv_header)

    # Load labels        results.append(obj)

    labels_matrix, video_files, neural_files = dataset.load_labels_file(labels_file)

    # Optionally limit samples in the dataset, this is used in debugging to reduce training time
    if limit_samples is not None:
        labels_matrix = labels_matrix[-limit_samples:]

    # Compute predictions
    # model = ModelSleepState()
    # model.init()
    output = inference_on_dataset(
        labels_matrix=labels_matrix,
        video_files=video_files,
        neural_files=neural_files,
        neural_files_basepath=neural_files_basepath,
        sample_width=sample_width,
        n_channels=n_channels,
        load_model=load_model,
        corrupt_neural_files=corrupt_neural_files,
        include_video_files=include_video_files,
        local_cache_size=local_cache_size,
        local_cache_workers=local_cache_workers,
        include_channels=include_channels,
        hp_max_predict_time_fps=hp_max_predict_time_fps,
    )
    write_result(output_file=output_file_tmp, output=output, video_files=video_files, neural_files=neural_files)

    # Zip results, move them to their final location, clean up
    output_file_zip = output_file_tmp + '.zip'
    with smart_open.open(output_file_zip, 'wb') as f:
        with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(output_file_tmp, arcname=os.path.basename(output_file))
    tf.io.gfile.copy(output_file_zip, output_file + '.zip', overwrite=True)
    tf.io.gfile.copy(output_file_tmp, output_file, overwrite=True)  # removed .csv after making sure .zip works
    os.remove(output_file_tmp)
    os.remove(output_file_zip)
    print('Predictions CSV saved to: {}'.format(output_file + '.zip'))


def main(command: str = None, **kwargs):
    globals()[command](**kwargs)


if __name__ == '__main__':
    main(**arg_parser())
