""" Unittests for train.py """
from common_py_utils.test_utils import BaseTestCase
import train


class TestTrain(BaseTestCase):
    def setUp(self) -> None:
        self.modelparams = {
            'sample_width': 2,
            'include_channels': 'all',
            'batch_size': 1,
        }

    def test_main(self):

        train.train(self.trainingparams, self.modelparams, self.datasetparams)
        self.fail()
