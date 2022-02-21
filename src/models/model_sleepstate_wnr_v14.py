import warnings
warnings.filterwarnings('ignore')
import tensorflow.compat.v1 as tf
import numpy as np
import common_py_utils.model_base as model_base
from common_py_utils.common_utils import SimpleNamespace
from common_py_utils.model_base import FetchOp
from common_py_utils.model_base import FetchOpEval
import common_py_utils.common_utils as common_utils
import io
import plotly.graph_objs as go
tf.disable_eager_execution()


# noinspection DuplicatedCode
class ModelSleepStateWNR(model_base.ModelBase):
    """
    Version wnr_v14 evolves v13, tests using softmax cross entropy instead of sigmoid
    """
    def __init__(self,
                 session: tf.Session = None,
                 device: str = None,
                 checkpoint_period_min: float = 60,
                 scope: str = '',
                 dataset_obj=None,
                 # HYPER-PARAMETERS
                 batch_size: int = 1,
                 sample_width_before: int = None,
                 sample_width_after: int = None,
                 include_channels: (tuple, list) = None,
                 hp_max_predict_time_fps: int = None,  # max time in frames-per-second used to compute time_label
                 l2_regularization: float = None,
                 learning_rate: (float, dict) = None,
                 ) -> None:
        """

        :param session: tensorflow session, though this is a relic of an older way to structure the model and would be better removed in a refactoring of the code where ModelBase is eliminated and its functions turned to common utility static methods.
        :param device: optionally a device (string) where tensors should be placed.
        :param checkpoint_period_min: same issue as session
        :param scope: scopes all tensors, a string.
        :param dataset_obj: the dataset.DatasetBase object which contains some definitions the model may need to know about
        :param batch_size:
        :param sample_width_before:
        :param sample_width_after:
        :param include_channels:
        :param hp_max_predict_time_fps:
        :param l2_regularization:
        :param learning_rate: a fixed learning rate as a scalar float, or a piecewise constant defined with a dictionary of {'boundaries': [100000, 110000], 'values': [1.0, 0.5, 0.1]}, see example at https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/piecewise_constant.
        """
        super().__init__(session, device, checkpoint_period_min, scope)

        self.sample_width = sample_width_before + sample_width_after
        self.n_channels = dataset_obj.n_channels
        self.hp_max_predict_time_fps = hp_max_predict_time_fps
        self.batch_size = batch_size
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate

        # lazy init fetch_ops
        self._fetch_ops = None
        self._fetch_ops_evaluate = None

        self.include_channels = common_utils.parse_include_channels(include_channels)

    @property
    def fetch_ops(self):
        if self._fetch_ops is None:
            metrics_update_frequency = 200
            loss_update_frequency = 10
            plot_output_frequency = 10000

            self._fetch_ops = [
                # Training -----------------------------------------------------------------
                FetchOp(name='train_op', tensor=self.ops.train_op,
                        update_frequency=1, is_reported=False),
                FetchOp(name='metrics_update_op', tensor=self.ops.metrics_update_op,
                        update_frequency=1, is_reported=False),
                FetchOp(name='loss', tensor=self.ops.loss,
                        update_frequency=1, is_reported=False),
                # Metrics -----------------------------------------------------------------
                FetchOp(name='loss_metric', tensor=self.ops.loss,
                        update_frequency=loss_update_frequency, is_reported=True),
                FetchOp(name='reset_local_variables', tensor=self.ops.reset_local_variables,
                        update_frequency=metrics_update_frequency, is_reported=False),
                FetchOp(name='accuracy_wnr', tensor=self.ops.accuracy_wnr,
                        update_frequency=metrics_update_frequency, is_reported=True),
                FetchOp(name='error_rate_wnr_long_term_state', tensor=self.ops.error_rate_wnr_long_term_state,
                        update_frequency=metrics_update_frequency, is_reported=True),
                FetchOp(name='error_rate_wnr_short_term_state', tensor=self.ops.error_rate_wnr_short_term_state,
                        update_frequency=metrics_update_frequency, is_reported=True),
                FetchOp(name='all_weights_mean', tensor=self.ops.all_weights_mean,
                        update_frequency=metrics_update_frequency, is_reported=True),
                FetchOp(name='all_weights_std', tensor=self.ops.all_weights_std,
                        update_frequency=metrics_update_frequency, is_reported=True),
                FetchOp(name='l2_regularization', tensor=self.ops.l2_regularization,
                        update_frequency=metrics_update_frequency, is_reported=True),
                FetchOp(name='learning_rate', tensor=self.ops.learning_rate,
                        update_frequency=metrics_update_frequency, is_reported=True),
                # Debug -----------------------------------------------------------------
                FetchOp(name='x', tensor=self.ops.x,
                        update_frequency=plot_output_frequency, is_reported=False),
                FetchOp(name='label_wnr', tensor=self.ops.label_wnr,
                        update_frequency=plot_output_frequency, is_reported=False),
                FetchOp(name='predicted_wnr', tensor=self.ops.predicted_wnr,
                        update_frequency=plot_output_frequency, is_reported=False),
            ]

        return self._fetch_ops

    @property
    def fetch_ops_evaluate(self):
        """ Returns a list of FetchOpEval objects which defines which ops to output from tensorflow during evaluation """
        if self._fetch_ops_evaluate is None:
            self._fetch_ops_evaluate = [
                FetchOpEval(name='label_wnr_012', tensor=self.ops.label_wnr),
                FetchOpEval(name='predicted_wnr_012', tensor=self.ops.predicted_wnr),
                FetchOpEval(name='confidence_wnr_01', tensor=self.ops.confidence_wnr),
                FetchOpEval(name='label_time', tensor=self.ops.input_tensors['label_time']),
                # FetchOpEval(name='hp_max_predict_time_fps', tensor=self.ops.input_tensors['hp_max_predict_time_fps']),  # todo, this should be in wnr-15 and onward and reported in summary stats output
                FetchOpEval(name='probability_wake', tensor=self.ops.output_probability_dist[:, 0]),
                FetchOpEval(name='probability_nrem', tensor=self.ops.output_probability_dist[:, 1]),
                FetchOpEval(name='probability_rem', tensor=self.ops.output_probability_dist[:, 2]),
                FetchOpEval(name='video_filename_ix', tensor=self.ops.input_tensors['video_filename_ixs']),
                FetchOpEval(name='video_filename', tensor=self.ops.input_tensors['video_filenames']),
                FetchOpEval(name='video_frame_ix', tensor=self.ops.input_tensors['video_frame_offsets']),
                FetchOpEval(name='neural_filename_ix', tensor=self.ops.input_tensors['neural_filename_ixs']),
                FetchOpEval(name='neural_filename', tensor=self.ops.input_tensors['neural_filenames']),
                FetchOpEval(name='neural_offset', tensor=self.ops.input_tensors['neural_offsets']),
            ]

        return self._fetch_ops_evaluate

    def status_message(self, out: SimpleNamespace):
        """ Generates an appropriate status message for this model. """
        return 'loss {:.3f}'.format(out.loss)

    def build_model(self, input_tensors: dict) -> None:
        assert self.hp_max_predict_time_fps is not None, 'Required parameter'

        with tf.variable_scope(self.scope), tf.device(self.device):
            # Input tensors
            self.ops.input_tensors = input_tensors
            input_neural = input_tensors['neural_data']
            neural_data_offsets = input_tensors['neural_data_offsets']
            sleep_state_int = input_tensors['sleep_states']
            label_time = input_tensors['label_time']

            assert self.include_channels is None or (isinstance(self.include_channels, (tuple, list)) and len(self.include_channels) > 0)
            assert self.sample_width is not None
            assert self.n_channels is not None
            assert isinstance(input_neural, str) or input_neural.shape.dims is not None, 'Input tensor shape is not known.'

            self.ops.global_step = tf.train.get_or_create_global_step()

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

            assert x.dtype == tf.int16
            x = tf.cast(x, dtype=tf.float32)
            x = tf.multiply(x, 0.00001)

            # batch_size, segment_length, n_channels = x.shape

            self.ops.cnn_output = self.build_cnn(self.ops, x)

            self.ops.output = tf.keras.layers.Dense(units=3)(self.ops.cnn_output)
            self.ops.output_probability_dist = tf.nn.softmax(self.ops.output)

            self.ops.loss = self.build_loss(
                self.ops, self.ops.output, sleep_state_int, self.l2_regularization
            )

            self.ops.train_op = self.build_optimizer(self.ops, self.ops.loss, self.learning_rate)

            self.build_metrics(self.ops, label_time, self.hp_max_predict_time_fps)

            self.ops.reset_local_variables = tf.local_variables_initializer()

    @staticmethod
    def build_optimizer(ops, loss, learning_rate):
        learning_rate = 1.0 if learning_rate is None else learning_rate

        ops.learning_rate = tf.constant(learning_rate, dtype=tf.float32) \
            if isinstance(learning_rate, float) \
            else tf.train.piecewise_constant(tf.train.get_or_create_global_step(),
                                             boundaries=learning_rate['boundaries'],
                                             values=learning_rate['values']) \
            if isinstance(learning_rate, dict) \
            else None

        train_op = tf.train.AdamOptimizer(learning_rate=ops.learning_rate).minimize(
            loss=loss, global_step=tf.train.get_or_create_global_step()
        )

        return train_op

    @staticmethod
    def build_metrics(ops, label_time, hp_max_predict_time_fps):
        # Compute metrics
        ops.accuracy_wnr, accuracy_wnr_update_op = tf.metrics.accuracy(
            labels=ops.label_wnr, predictions=ops.predicted_wnr, name='accuracy_wnr',
        )

        # WAKE|NREM|REM error rate for samples > 30 sec from a transition
        # hp_max_predict_time_fps determines the range for label_time in [-1, +1], assuming a frame rate of 15 fps
        long_term_state = (30 * 15) / hp_max_predict_time_fps  # 30 sec @ 15 fps over scale used to produce label_time
        is_long_term_state = tf.abs(label_time) > long_term_state
        accuracy_long_term_state, accuracy_long_term_state_update_op = tf.metrics.accuracy(
            labels=ops.label_wnr,
            predictions=ops.predicted_wnr,
            weights=tf.cast(is_long_term_state, tf.float32),
            name='accuracy_long_term_state',
        )
        accuracy_short_term_state, accuracy_short_term_state_update_op = tf.metrics.accuracy(
            labels=ops.label_wnr,
            predictions=ops.predicted_wnr,
            weights=1.0 - tf.cast(is_long_term_state, tf.float32),
            name='accuracy_short_term_state',
        )
        ops.error_rate_wnr_long_term_state = 1 - accuracy_long_term_state  # save error rate rather than accuracy
        ops.error_rate_wnr_short_term_state = 1 - accuracy_short_term_state  # save error rate rather than accuracy

        pred_vector_length = tf.norm(ops.output_probability_dist, axis=1)  # vector length (assuming this is length 3)
        chance_vector_length = tf.sqrt((1/3)**2 * 3)  # fixed for Wake|NREM|REM
        ops.confidence_wnr = (pred_vector_length - chance_vector_length) / (1 - chance_vector_length)  # rescale [0, 1]

        ops.metrics_update_op = tf.group(
            accuracy_wnr_update_op, accuracy_long_term_state_update_op, accuracy_short_term_state_update_op
        )

    @staticmethod
    def build_cnn(ops, input_neural, name_scope='ops_timequantcnn', variable_scope='vars_timequantcnn') -> tf.Tensor:
        # Expected input shape: [batch, segment_length, channels]

        msra_initializer = tf.initializers.variance_scaling(scale=2.0, mode='fan_in', distribution='normal')

        batch_size, segment_length, n_channels = input_neural.shape

        # Create filters
        with tf.variable_scope(variable_scope):
            filters0 = tf.get_variable(name='conv0_filters', shape=(30, n_channels, 320), initializer=msra_initializer)
            filters1 = tf.get_variable(name='conv1_filters', shape=(30, 320, 384), initializer=msra_initializer)
            filters2 = tf.get_variable(name='conv2_filters', shape=(30, 384, 448), initializer=msra_initializer)
            filters3 = tf.get_variable(name='conv3_filters', shape=(30, 448, 512), initializer=msra_initializer)
            filters4 = tf.get_variable(name='conv4_filters', shape=(30, 512, 512), initializer=msra_initializer)
            filters5 = tf.get_variable(name='conv5_filters', shape=(30, 512, 512), initializer=msra_initializer)
            filters6 = tf.get_variable(name='conv6_filters', shape=(30, 512, 512), initializer=msra_initializer)
            filters7 = tf.get_variable(name='conv7_filters', shape=(30, 512, 512), initializer=msra_initializer)

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters0)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters1)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters2)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters3)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters4)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters5)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters6)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters7)

        # Create conv operations
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
            x = tf.nn.relu(x, name='conv7_relu')

        # compute mean and variance of weights for reporting purposes
        filters_flat = tf.concat(
            values=[tf.reshape(filterN, (-1,))for filterN in [filters0, filters1, filters2, filters3,
                                                              filters4, filters5, filters6, filters7]],
            axis=0
        )
        ops.all_weights_mean = tf.math.reduce_mean(filters_flat)
        ops.all_weights_std = tf.math.reduce_std(filters_flat)

        # validate that the convolutions produced a vector of outputs, if not the reshape operation that follows would change the batch dimension erroneously
        shape_assert = tf.debugging.assert_equal(x.shape[1], 1)
        with tf.control_dependencies([shape_assert]):
            x = tf.reshape(x, shape=(-1, 512))

        return x

    @staticmethod
    def build_loss(ops, output, sleep_state_int, l2_regularization):
        # # todo this doesn't support 5-state as-is
        # sleep_state_zero_base = tf.cast(tf.subtract(sleep_state_int, 1), tf.int64)
        # sleep_state_onehot = tf.one_hot(indices=sleep_state_zero_base, depth=3, dtype=tf.float32)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=sleep_state_onehot, logits=output, name='loss')
        # loss = tf.math.reduce_mean(loss, name='reduce_mean')
        #
        # if l2_regularization is not None:
        #     ops.l2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.WEIGHTS)]) * l2_regularization
        #     loss = loss + ops.l2_regularization
        #
        # ops.label_wnr = sleep_state_zero_base
        # ops.predicted_wnr = tf.math.argmax(output, axis=1)

        is_wake = tf.logical_or(tf.equal(sleep_state_int, 1), tf.equal(sleep_state_int, 5))
        is_nrem = tf.equal(sleep_state_int, 2)
        is_rem = tf.equal(sleep_state_int, 3)
        labels_onehot_wnr = tf.cast(tf.stack((is_wake, is_nrem, is_rem), axis=1), dtype=tf.float32, name='labels_wnr')

        loss_wnr = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels_onehot_wnr, logits=output, name='loss_wnr'
        )
        loss_wnr = tf.reduce_mean(loss_wnr, name='loss_wnr')

        label_wnr = tf.argmax(labels_onehot_wnr, axis=1, name='label_wnr')
        predicted_wnr = tf.argmax(output, axis=1, name='predicted_wnr')

        # account for the possibility that labels are -1 when unset for eval only
        label_wnr = tf.where(
            tf.math.equal(sleep_state_int, -1),
            x=tf.ones_like(sleep_state_int, dtype=tf.int64) * -1,
            y=label_wnr,
        )

        if l2_regularization is not None:
            ops.l2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.WEIGHTS)]) * l2_regularization
            loss_wnr = loss_wnr + ops.l2_regularization

        ops.label_wnr = label_wnr
        ops.predicted_wnr = predicted_wnr
        return loss_wnr

    def log_image(self, out: SimpleNamespace):
        """
        Receives the result from tensorflow (in a SimpleNamespace object) each step and returns a
        list of one or more images to log to comet.ml as a tuple of (title, image) pairs.

        :param out: The output from tensorflow as defined by fetch_ops()
        :return: a list of one or more images, as a tuple of (image, byte-stream) to be logged to comet.ml or None
        """
        if hasattr(out, 'x'):
            chan = np.random.randint(0, out.x.shape[2])
            print('DEBUG> log_image, global step {}, channel {}'.format(out.global_step, chan))

            fig = go.Figure(
                data=[
                    go.Scatter(y=out.x[0, :, chan])
                ],
                layout=go.Layout(
                    title='Sample data: filename {f}, sample_offset {o}, channel {c}'.format(f='todo', o='todo', c=chan)
                )
            )

            fig_bytes = fig.to_image(format='png')
            buf = io.BytesIO(fig_bytes)

            correct_prediction = out.label_wnr == out.predicted_wnr
            title = '{yesno}-label{label}-pred{pred}'.format(
                yesno='Y' if correct_prediction else 'N',
                label=out.label_wnr,
                pred=out.predicted_wnr
            )

            return [(title, buf)]

    def update_callback(self, out: SimpleNamespace):
        """ Generic callback made on each update step. Useful for debugging or other generic purposes. """
        pass
