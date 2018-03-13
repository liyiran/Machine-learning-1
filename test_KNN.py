from unittest import TestCase, skip
from hw1_knn import KNN
from data import generate_data_cancer
import numpy.testing
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, normalize, min_max_scale
from utils import f1_score
from utils import NormalizationScaler, MinMaxScaler

class TestKNN(TestCase):
    def setUp(self):
        self.knn = KNN(1, euclidean_distance)

    def test_train(self):
        features = [[1, 1], [1, 2], [2, 2], [9, 9], [8, 8], [8, 9]]
        values = [0, 0, 0, 1, 1, 1]
        self.knn.train(features, values)
        point1 = [0, 0]
        neighbor = self.knn.get_neighbors(point1)
        self.assertEqual(0, self.knn.get_response(neighbor))
        numpy.testing.assert_array_equal(numpy.array([[1, 1, 0]]), neighbor)
        point2 = [10, 10]
        neighbor = self.knn.get_neighbors(point2)
        numpy.testing.assert_array_equal(numpy.array([[9, 9, 1]]), neighbor)
        self.assertEqual(1, self.knn.get_response(neighbor))

    @skip("clarify inner product")
    def test_inner_product_knn(self):
        knn = KNN(1, inner_product_distance)
        features = [[1, 1], [1, 2], [2, 2], [9, 9], [8, 8], [8, 9]]
        values = [0, 0, 0, 1, 1, 1]
        knn.train(features, values)
        point1 = [0, 0]
        neighbor = knn.get_neighbors(point1)
        self.assertEqual(0, knn.get_response(neighbor))
        numpy.testing.assert_array_equal(numpy.array([[1, 1, 0]]), neighbor)
        point2 = [10, 10]
        neighbor = knn.get_neighbors(point2)
        numpy.testing.assert_array_equal(numpy.array([[1, 1, 0]]), neighbor)
        self.assertEqual(1, knn.get_response(neighbor))

    def test_predict(self):
        features = [[1, 1], [1, 2], [2, 2], [9, 9], [8, 8], [8, 9]]
        values = [0, 0, 0, 1, 1, 1]
        self.knn.train(features, values)
        points = [[0, 0], [10, 10]]
        self.assertListEqual([0, 1], self.knn.predict(points))

    def test_knn(self):
        features, labels = generate_data_cancer()
        train_features, train_labels = features[:400], labels[:400]
        valid_features, valid_labels = features[400:460], labels[400:460]
        test_features, test_labels = features[460:], labels[460:]
        assert len(train_features) == len(train_labels) == 400
        assert len(valid_features) == len(valid_labels) == 60
        assert len(test_features) == len(test_labels) == 109
        distance_funcs = {
            # 'euclidean': euclidean_distance,
            # 'gaussian': gaussian_kernel_distance,
            'inner_prod': inner_product_distance,
        }
        for name, func in distance_funcs.items():
            best_f1_score, best_k = -1, 0
            for k in [1]:
                model = KNN(k=k, distance_function=func)
                model.train(train_features, train_labels)
                # print(train_labels)
                # print(model.predict(train_features))
                train_f1_score = f1_score(train_labels, model.predict(train_features))
                valid_f1_score = f1_score(valid_labels, model.predict(valid_features))
                print(f'[part 2.1] {name}\tk: {k:d}\t' f'train: {train_f1_score:.5f}\t' f'valid: {valid_f1_score:.5f}')
                if valid_f1_score > best_f1_score:
                    best_f1_score, best_k = valid_f1_score, k
        model = KNN(k=best_k, distance_function=func)
        model.train(train_features + valid_features,
                    train_labels + valid_labels)
        test_f1_score = f1_score(test_labels, model.predict(test_features)
                                 )
        print()
        print(f'[part 2.1] {name}\tbest_k: {best_k:d}\t' f'test f1 score: {test_f1_score:.5f}')
        print()

    def test_normalization(self):
        scaling_functions = {
            'min_max_scale': MinMaxScaler,
            'normalize': NormalizationScaler,
        }
        distance_funcs = {
            'euclidean': euclidean_distance,
            'gaussian': gaussian_kernel_distance,
            'inner_prod': inner_product_distance,
        }
        features, labels = generate_data_cancer()
        train_features, train_labels = features[:400], labels[:400]
        valid_features, valid_labels = features[400:460], labels[400:460]
        test_features, test_labels = features[460:], labels[460:]
        assert len(train_features) == len(train_labels) == 400
        assert len(valid_features) == len(valid_labels) == 60
        assert len(test_features) == len(test_labels) == 109
        for scaling_name, scaling_class in scaling_functions.items():
            for name, func in distance_funcs.items():
                scaler = scaling_class()
                train_features_scaled = scaler(train_features)
                valid_features_scaled = scaler(valid_features)

                best_f1_score, best_k = 0, -1
                for k in [1, 3, 10, 20, 50]:
                    model = KNN(k=k, distance_function=func)
                    model.train(train_features_scaled, train_labels)
                    train_f1_score = f1_score(
                        train_labels, model.predict(train_features_scaled))

                    valid_f1_score = f1_score(
                        valid_labels, model.predict(valid_features_scaled))
                    print('[part 2.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

                    if valid_f1_score > best_f1_score:
                        best_f1_score, best_k = valid_f1_score, k

                # now change it to new scaler, since the training set changes
                scaler = scaling_class()
                combined_features_scaled = scaler(train_features + valid_features)
                test_features_scaled = scaler(test_features)

                model = KNN(k=best_k, distance_function=func)
                model.train(combined_features_scaled, train_labels + valid_labels)
                test_f1_score = f1_score(test_labels, model.predict(test_features_scaled))
                print()
                print('[part 2.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
                      'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
                print()
