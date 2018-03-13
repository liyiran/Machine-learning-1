from unittest import TestCase, skip

import matplotlib.pyplot as plt
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal

from data import generate_data_part_1, generate_data_part_2, generate_data_part_3
from hw1_lr import LinearRegression, LinearRegressionWithL2Loss
from utils import mean_squared_error, polynomial_features


class TestLinearRegression(TestCase):
    def setUp(self):
        self.linear_regression = LinearRegression(1)

    def test_train(self):
        x = [[63], [64], [66], [69], [69], [71], [71], [72], [73], [75]]
        y = [127, 121, 142, 157, 162, 156, 169, 165, 181, 208]
        self.linear_regression.train(x, y)
        weights = self.linear_regression.get_weights()
        assert_array_almost_equal([-266.53, 6.13], weights, decimal=2)

    def test_predict(self):
        x = [[63], [64], [66], [69], [69], [71], [71], [72], [73], [75]]
        y = [127, 121, 142, 157, 162, 156, 169, 165, 181, 208]
        self.linear_regression.train(x, y)
        assert_array_almost_equal([120.13, 126.27, 138.55, 156.96, 156.96, 169.23, 169.23,
                                   175.37, 181.51, 193.78],
                                  self.linear_regression.predict([[63], [64], [66], [69], [69], [71], [71], [72], [73], [75]]), decimal=2)

    def test_add_one_ahead(self):
        a = numpy.array([[1, 1], [2, 2], [3, 3]])
        b = numpy.insert(a, 0, 1, axis=1)
        assert_array_equal(numpy.array([[1, 1, 1], [1, 2, 2], [1, 3, 3]]), b)
        assert_array_equal(numpy.array([[1, 1], [2, 2], [3, 3]]), a)

    def test_lr_integration(self):
        features, values = generate_data_part_1()
        model = LinearRegression(nb_features=1)
        model.train(features, values)
        mse = mean_squared_error(values, model.predict(features))
        self.assertAlmostEqual(0.00175, mse, places=5)
        plt.scatter([x[0] for x in features], values, label='origin');
        plt.plot([x[0] for x in features], model.predict(features), label='predicted');
        plt.title("Holy shit")
        plt.legend()
        plt.show()

    def test_lr_integration_l2(self):
        features, values = generate_data_part_1()
        model = LinearRegressionWithL2Loss(nb_features=1, alpha=0.0)
        model.train(features, values)
        mse = mean_squared_error(values, model.predict(features))
        # self.assertAlmostEqual(0.00175, mse, places=5)
        plt.scatter([x[0] for x in features], values, label='origin');
        plt.plot([x[0] for x in features], model.predict(features), label='predicted');
        plt.legend()

    def test_feature_select(self):
        features, values = generate_data_part_2()
        model = LinearRegression(nb_features=1)
        model.train(features, values)
        mse = mean_squared_error(values, model.predict(features))
        print(f'[part 1.3.1]\tmse: {mse:.5f}')
        plt.scatter([x[0] for x in features], values, label='origin');
        plt.plot([x[0] for x in features], model.predict(features), label='pre dicted');
        plt.legend()
        plt.show()

    def test_add_polynomial_feature(self):
        features, values = generate_data_part_2()
        plt.scatter([x[0] for x in features], values, label='origin');
        for k in [2, 4, 10]:
            # TODO: confirm polynomial feature and k = Xk
            features_extended = polynomial_features(features, k)
            model = LinearRegression(nb_features=k)
            model.train(features_extended, values)
            mse = mean_squared_error(values, model.predict(features_extended))
            print(f'[part 1.3.2]\tk: {k:d}\tmse: {mse:.5f}')
            plt.plot([x[0] for x in features], model.predict(features_extended), label=f'k={k}');
            plt.legend()
            plt.show()

    def test_data_procession(self):
        features, values = generate_data_part_3()
        train_features, train_values = features[:100], values[:100]
        valid_features, valid_values = features[100:120], values[100:120]
        test_features, test_values = features[120:], values[120:]
        assert len(train_features) == len(train_values) == 100
        assert len(valid_features) == len(valid_values) == 20
        assert len(test_features) == len(test_values) == 30
        best_mse, best_k = 1e10, -1
        for k in [1, 3, 10]:
            train_features_extended = polynomial_features(train_features, k)
            model = LinearRegression(nb_features=k)
            model.train(train_features_extended, train_values)
            train_mse = mean_squared_error(train_values, model.predict(train_features_extended))
            valid_features_extended = polynomial_features(valid_features, k)
            valid_mse = mean_squared_error(valid_values, model.predict(valid_features_extended))
            print(f'[part 1.4.1]\tk: {k:d}\t'
                  f'train mse: {train_mse:.5f}\tvalid mse: {valid_mse:.5f}')
            if valid_mse < best_mse:
                best_mse, best_k = valid_mse, k
        # combined_features_extended = polynomial_features(train_features + test_features, best_k)
        # model = LinearRegression(nb_features=best_k)
        # model.train(combined_features_extended, train_values + test_values)
        # test_features_extended = polynomial_features(test_features, best_k) 
        # test_mse = mean_squared_error(test_values, model.predict(test_features_extended))
        # print(f'[part 1.4.1 Linear Regression]\tbest_k: {best_k:d}\ttest mse: {test_mse:.5f}')
