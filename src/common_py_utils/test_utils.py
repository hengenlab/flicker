import unittest
import numpy as np
import math
import logging
import sys


class BaseTestCase(unittest.TestCase):
    def assert_complex_equal(self, d1: object, d2: object, ndarray_tolerance=None):
        """ Recursive asserts with numpy special cases handled. """
        if isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
            self.assertEqual(len(d1), len(d2))
            for d1_item, d2_item in zip(d1, d2):
                self.assert_complex_equal(d1_item, d2_item)
        elif isinstance(d1, dict) and isinstance(d2, dict):
            d1_keys = sorted(d1.keys())
            d2_keys = sorted(d2.keys())
            self.assertCountEqual(d1_keys, d2_keys)
            for d1_key, d2_key in zip(d1_keys, d2_keys):
                self.assert_complex_equal(d1[d1_key], d2[d2_key])
        elif isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
            if ndarray_tolerance is not None:
                np.testing.assert_allclose(d1, d2, atol=ndarray_tolerance)
            else:
                np.testing.assert_equal(d1, d2)
        elif isinstance(d1, float) and isinstance(d2, float) and math.isnan(d1) and math.isnan(d2):
            pass
        else:
            self.assertEqual(d1, d2)
