from unittest import TestCase
from utils import NormalizationScaler, MinMaxScaler
import numpy
from numpy.testing import assert_array_almost_equal


class TestNormalizationScaler(TestCase):
    def test_normalization(self):
        normal = NormalizationScaler()
        features = [[3, 4], [1, -1], [0, 0]]
        assert_array_almost_equal(numpy.array([[0.6, 0.8], [0.707107, -0.707107], [0, 0]]), numpy.array(normal(features)))
        assert_array_almost_equal(numpy.array([[0.6, 0.8], [0.707107, -0.707107], [0, 0]]), numpy.array(normal(features)))

    def test_min_max_scalar(self):
        mim_max = MinMaxScaler()
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        train_features_scaled = mim_max(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        self.assertListEqual([[0, 1], [1, 0]], train_features_scaled)

        test_features_scaled = mim_max(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        self.assertListEqual([[10, 0.1]], test_features_scaled)

        new_scaler = MinMaxScaler()  # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]])  # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
        self.assertListEqual([[20, 1]], test_features_scaled)
