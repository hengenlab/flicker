import unittest
import model_sleepstate_generate_labels
import numpy as np


# noinspection PyPep8Naming
class TestModuleFunctions(unittest.TestCase):
    def test_compute_time_from_last_label_int8_overflow_bug(self):
        test_data = np.array([1] + [2] * 200 + [1], dtype=np.int8)
        result_wF, result_wB = model_sleepstate_generate_labels.compute_time_from_last_label(test_data, [1, 5])
        expected_result_wF = np.array([-1] + list(range(200, 0, -1)) + [-1])
        expected_result_wB = np.array([-1] + list(range(1, 201)) + [-1])
        np.testing.assert_equal(expected_result_wF, result_wF)
        np.testing.assert_equal(expected_result_wB, result_wB)

    def test_compute_time_from_last_label(self):
        test_data = np.array([2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 1, 1, 2, 2, 5, 5, 3, 3])

        # wake state, forward and backward
        expected_result_wF = np.array([1, -1, -1, -1, -1, 6, 5, 4, 3, 2, 1, -1, -1, -1, 2, 1, -1, -1, -2, -2])
        expected_result_wB = np.array([-2, -1, -1, -1, -1, 1, 2, 3, 4, 5, 6, -1, -1, -1, 1, 2, -1, -1, 1, 2])

        result_wF, result_wB = model_sleepstate_generate_labels.compute_time_from_last_label(test_data, [1, 5])

        np.testing.assert_equal(expected_result_wF, result_wF)
        np.testing.assert_equal(expected_result_wB, result_wB)

        # nrem state, forward and backward
        expected_result_nF = np.array([-1, 4, 3, 2, 1, -1, -1, -1, 6, 5, 4, 3, 2, 1, -1, -1, -2, -2, -2, -2])
        expected_result_nB = np.array([-1, 1, 2, 3, 4, -1, -1, -1, 1, 2, 3, 4, 5, 6, -1, -1, 1, 2, 3, 4])

        result_nF, result_nB = model_sleepstate_generate_labels.compute_time_from_last_label(test_data, [2])

        np.testing.assert_equal(expected_result_nF, result_nF)
        np.testing.assert_equal(expected_result_nB, result_nB)

        # rem state, forward and backward
        expected_result_rF = np.array([8, 7, 6, 5, 4, 3, 2, 1, -1, -1, 8, 7, 6, 5, 4, 3, 2, 1, -1, -1])
        expected_result_rB = np.array([-2, -2, -2, -2, -2, -2, -2, -2, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1])

        result_rF, result_rB = model_sleepstate_generate_labels.compute_time_from_last_label(test_data, [3])

        np.testing.assert_equal(expected_result_rF, result_rF)
        np.testing.assert_equal(expected_result_rB, result_rB)
