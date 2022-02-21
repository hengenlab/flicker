import unittest
import common_py_utils.common_utils as common_utils
from common_py_utils.test_utils import BaseTestCase
from multiprocessing import shared_memory
import pickle
import numpy as np
from fs import tempfs


class TestSharedMemory(BaseTestCase):
    def setUp(self) -> None:
        self.test_obj_a = ['a', 'b', 42]
        self.test_obj_b = 42
        self.test_obj_c = np.array([1, 2, 3], dtype=np.int32)

    def test_create_shared_memory(self, unlink=True):
        pickle_obj_a = pickle.dumps(self.test_obj_a)
        pickle_obj_b = pickle.dumps(self.test_obj_b)
        pickle_obj_c = pickle.dumps({'length': self.test_obj_c.nbytes, 'shape': self.test_obj_c.shape, 'dtype': self.test_obj_c.dtype})

        expected_result = b'\x00' + len(pickle_obj_a).to_bytes(4, byteorder='little') + pickle_obj_a + \
                          b'\x00' + len(pickle_obj_b).to_bytes(4, byteorder='little') + pickle_obj_b + \
                          b'\x01' + len(pickle_obj_c).to_bytes(4, byteorder='little') + pickle_obj_c + self.test_obj_c.tobytes()

        shm_name = common_utils.create_shared_memory(self.test_obj_a, self.test_obj_b, self.test_obj_c)
        shm = shared_memory.SharedMemory(name=shm_name)
        result = shm.buf

        self.assertEqual(expected_result, result)
        shm.close()
        if unlink:
            shm.unlink()

        return shm_name

    def test_deserialize_shared_memory(self):
        expected_results = (self.test_obj_a, self.test_obj_b, self.test_obj_c)

        name = self.test_create_shared_memory(unlink=False)
        results = common_utils.deserialize_shared_memory(name=name)

        self.assert_complex_equal(expected_results, results)

    def test_zero_dim_array_bug(self):
        """ A bug occurred with 0D ndarrays, reproduced here prior to correcting it. """
        test_array = np.array(42, dtype=np.int64)
        shm_name = common_utils.create_shared_memory('abc', test_array, 'xyz')
        deserialized_test_data = common_utils.deserialize_shared_memory(shm_name)

        expected = ('abc', test_array, 'xyz')

        self.assertEqual(expected, deserialized_test_data)


class TestChoiceWoReplace(BaseTestCase):
    def test_choice_wo_replace(self):
        inp = np.array([1, 2, 3])

        np.random.seed(42)
        result = common_py_utils.choice_wo_replace(inp, size=11)

        # Validate that each repeat of the input is a full permutation without repeating values
        for i in range(3):
            uniques = result[i * 3:i * 3 + 3]
            self.assertTrue(len(uniques) == 3)

        # Validate that each repeat set of the input is permuted, depending on the seed this could be true by chance
        # so pick a seed where it's false so we can validate that we didn't miss permuting the repeated sets
        self.assertFalse(np.all(result[0:3] == result[3:6]))

        # Validate that the remainder is sampled wo replacement
        uniques = result[9:]
        self.assertTrue(len(uniques) == 2)


class TestFileFunctions(BaseTestCase):

    def test_file_list(self):
        with tempfs.TempFS() as tfs:
            tfs.touch('/file1')
            tfs.touch('/file2')
            tfs.touch('/file3')

            files = common_utils.file_list(tfs.getsyspath('/'))
            expected = ['file1', 'file2', 'file3']

            self.assert_complex_equal(expected, files)
