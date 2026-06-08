"""Regression tests for H6 #492: NaN/Inf validation on embeddings before insert.

serialize_f32() is the single choke point for all vector insertions.
It must reject NaN and Inf values to prevent corrupted embeddings from
entering the database.
"""

import math
import unittest

import numpy as np


class TestSerializeF32NaNValidation(unittest.TestCase):
    """serialize_f32 must reject NaN/Inf embeddings."""

    def test_rejects_nan_in_list(self):
        from truememory.vector_search import serialize_f32
        with self.assertRaises(ValueError):
            serialize_f32([0.1, float("nan"), 0.3])

    def test_rejects_inf_in_list(self):
        from truememory.vector_search import serialize_f32
        with self.assertRaises(ValueError):
            serialize_f32([0.1, float("inf"), 0.3])

    def test_rejects_neg_inf_in_list(self):
        from truememory.vector_search import serialize_f32
        with self.assertRaises(ValueError):
            serialize_f32([0.1, float("-inf"), 0.3])

    def test_rejects_nan_in_numpy(self):
        from truememory.vector_search import serialize_f32
        arr = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with self.assertRaises(ValueError):
            serialize_f32(arr)

    def test_rejects_inf_in_numpy(self):
        from truememory.vector_search import serialize_f32
        arr = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        with self.assertRaises(ValueError):
            serialize_f32(arr)

    def test_accepts_valid_embedding(self):
        from truememory.vector_search import serialize_f32
        result = serialize_f32([0.1, 0.2, 0.3])
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), 12)  # 3 floats * 4 bytes

    def test_accepts_valid_numpy(self):
        from truememory.vector_search import serialize_f32
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = serialize_f32(arr)
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), 12)

    def test_accepts_zeros(self):
        from truememory.vector_search import serialize_f32
        result = serialize_f32([0.0, 0.0, 0.0])
        self.assertIsInstance(result, bytes)


if __name__ == "__main__":
    unittest.main()
