""" Base class for TF 1.15 and below models. """
import comet_ml
import tensorflow.compat.v1 as tf
import numpy as np
import dataset
import common_py_utils.growing_epochs as growing_epochs
import types
from datetime import datetime
import fs.tempfs
import os
import socket
import common_py_utils.common_utils as common_utils
from recordtype import recordtype
from common_py_utils.common_utils import SimpleNamespace


# a tuple to store the details of each fetch_op in the Model used by property fetch_ops
# Property descriptions:
#   name - name of the tensor.
#   tensor - tensorflow tensor object.
#   update_frequency - How often to request this tensor during the training loop, 1 requests it every training step
#                      which would be necessary for the train and metric updates ops, but less frequent is common
#                      for loss and accuracy metrics.
#   is_reported - boolean specifying if the result should be reported to comet.ml
FetchOp = recordtype(
    'FetchOp', ['name', 'tensor', 'update_frequency', 'is_reported']
)
FetchOpEval = recordtype(
    'FetchOpEval', ['name', 'tensor']
)


class ModelBase:
    def __init__(self, session: tf.Session = None,
                 device: str = None,
                 checkpoint_period_min: float = 20,
                 scope: str = ''):
        # Public model components, access ops via (for example) self.ops.loss
        self.ops = types.SimpleNamespace()
        self.scope = scope

        self._session = session
        self._tf_saver = None
        if device is None:
            assert len(tf.config.list_logical_devices('GPU')) <= 1, 'Multiple GPUs found but device not specified.'
            self.device = device if device is not None \
                else 'GPU:0' if tf.test.is_gpu_available(cuda_only=True) else 'CPU'
        else:
            self.device = device

        self.checkpoint_period_min = checkpoint_period_min
        self.checkpoint_last_time = datetime.now()
        self.checkpoint_start_time = datetime.now()

    @property
    def fetch_ops(self):
        """
        Returns a list of [('name', tensor, step_frequency), ...] tuples. The tensor is from self.ops and the
        step_frequency is how often it should be fetched.

        This is a default implementation, this function should be overridden by the extending class.
        """
        ops = [
            FetchOp('train_op', self.ops.train_op, 1, False),
            FetchOp('loss', self.ops.loss, 1, False),
        ]
        return ops

    # noinspection PyMethodMayBeStatic
    def status_message(self, out: SimpleNamespace):
        """ Generates an appropriate status message for this step, may be None to not output at this step. """
        # todo produce the optional status message for this model
        status = ''
        return status

    @property
    def session(self):
        if self._session is None:
            self._session = create_session()
        return self._session

    @property
    def tf_saver(self):
        if self._tf_saver is None:
            # Remove the device which was added for multi-gpu support
            var_scope_map = {v.name.split('/', 1)[-1].split(':')[0]: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)} \
                if tf.train.get_or_create_global_step().name != 'global_step:0' else None
            self._tf_saver = tf.train.Saver(var_list=var_scope_map, max_to_keep=50)
        return self._tf_saver

    def export_model(self, reset=True):
        """ Returns a data structure of the model variables which can be restored, calls tf.reset_default_graph(). """
        var_value_map = {v.name: self.session.run(v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        if reset:
            if self._session is not None:
                self._session.close()
                self._session = None
            tf.reset_default_graph()
            self.ops = types.SimpleNamespace()
        return var_value_map

    def import_model(self, variable_names_and_values):
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            v.load(value=variable_names_and_values[v.name], session=self.session)

    def load_model(self, model_filename):
        files = common_utils.file_list(model_filename)
        files = [f for f in files if f.endswith('.meta')]

        assert len(files) > 0, f'Failed to validate that file(s) at: {model_filename}'
        file = os.path.join(os.path.dirname(model_filename), files[0][:-5])

        self.tf_saver.restore(self.session, file)
        print('Loaded model: {} at global_step {}'.format(
            file, self.session.run(tf.train.get_or_create_global_step())
        ))

    def init(self):
        """ Global variable initialization """
        tf.train.get_or_create_global_step()
        self.session.run(tf.local_variables_initializer())
        self.session.run(tf.global_variables_initializer())

    def save_checkpoint_periodically(self, checkpoint_dir, step, loss, force=False):
        """ Saves the model IFF it's been longer than self.checkpoint_period_min since the last save. """
        retval = None

        if checkpoint_dir is not None and \
                ((datetime.now() - self.checkpoint_last_time).total_seconds() > self.checkpoint_period_min * 60
                 or force):
            self.checkpoint_last_time = datetime.now()
            checkpoint_filepath = os.path.join(
                checkpoint_dir,
                '{hostname}-{timestamp}-step{step:06d}-loss{loss:.3f}'.format(
                    hostname=socket.gethostname(),
                    step=step,
                    timestamp=self.checkpoint_start_time.strftime('%Y-%m-%d-%Hh-%Mm-%Ss'),
                    loss=loss
                )
            )
            self.tf_saver.save(self.session, checkpoint_filepath)
            print('Saved checkpoint: {}'.format(checkpoint_filepath))
            retval = checkpoint_filepath

        return retval

    def shutdown(self):
        """ Shutdown is called at the end of processing to allow flush and cleanup operations. """

    def process_result(self, out: SimpleNamespace):
        """
        Accepts the output of the model as a SimpleNamespace.
        Typical use case is to write results to CSV.
        """

    def log_image(self, out: SimpleNamespace):
        """
        Accepts the output of the model as a SimpleNamespace. Return a list of tuples in the format:
        [ (name, image), (...), ... ]
        """
        pass

    def log_text(self, out: SimpleNamespace):
        """
        Accepts the output of the model as a SimpleNamespace. Return a list of text strings:
        [ string, ... ]
        """
        pass

    def update_callback(self, out: SimpleNamespace):
        """
        Called each update step, useful for debugging or other general purpose use.
        :return:
        """
        pass

    @staticmethod
    def sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))

    @staticmethod
    def softmax(x):
        z = x - np.max(x, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        softmax_val = numerator / denominator
        return softmax_val


def create_session():
    """ Creates a TF session with options. """
    # https://github.com/tensorflow/tensorflow/issues/23780
    # https://github.com/tensorflow/tensorflow/issues/5445
    from tensorflow.core.protobuf import rewriter_config_pb2
    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off

    session = tf.Session(config=config_proto)
    return session
