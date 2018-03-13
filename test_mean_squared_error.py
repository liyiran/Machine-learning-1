import math
from unittest import TestCase
import numpy
from random import randint
from numpy.testing import assert_array_almost_equal
from utils import polynomial_features, euclidean_distance, f1_score, gaussian_kernel_distance, inner_product_distance, mean_squared_error, min_max_scale, normalize


class TestMean_squared_error(TestCase):
    def test_mean_squared_error(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        result = mean_squared_error(y_true, y_pred)
        self.assertEqual(0.375, result)

    def test_f1_score(self):  # macro
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        self.assertEqual(0.8 / 3, f1_score(y_true, y_pred))

    def test_distance(self):
        a = [-1, 2, 3]
        b = [4, 0, -3]
        self.assertEqual(65, math.ceil(euclidean_distance(a, b) ** 2))

    def test_inner_product(self):
        a = [1, 2, 3]
        b = [0, 1, 2]
        self.assertEqual(6, inner_product_distance(a, b))

    def test_gaussian_kernal_distance(self):
        a = [5, 4, 2, 1]
        b = [1, 2, 4, 7]
        self.assertAlmostEqual(-9.3576229688401407e-14, gaussian_kernel_distance(a, b))

    def test_normalize(self):
        features = [[3, 4], [1, -1], [0, 0]]
        self.assertListEqual([[0.6, 0.8], [0.7071067811865475, -0.7071067811865475], [0, 0]], normalize(features))

    def test_min_max(self):
        features = [[2, -1], [-1, 5], [0, 0]]
        assert_array_almost_equal([[1, 0], [0, 1], [0.33333333, 0.16666667]], min_max_scale(features))

    def test_polynomial_features(self):
        features = [[1.0, 2, 3], [4.0, 5, 6], [7.0, 8, 9]]
        assert_array_almost_equal(numpy.array([[1.0, 2, 3, 1, 4, 9], 
                                               [4.0, 5, 6, 16, 25, 36], 
                                               [7.0, 8, 9, 49, 64, 81]]), numpy.array(polynomial_features(features, 2)))
