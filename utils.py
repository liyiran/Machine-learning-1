from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    local_sum = 0
    for i in range(len(y_true)):
        local_sum = local_sum + (y_true[i] - y_pred[i]) ** 2
    return local_sum / len(y_true)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    assert len(real_labels) == len(predicted_labels)
    label_set = set(real_labels).union(predicted_labels)
    f = 0
    for label in label_set:
        real = list(map(lambda l: l == label, real_labels))
        predict = list(map(lambda l: l == label, predicted_labels))
        f += f1_score_2_value(real, predict)  # TODO: confirm the average of f score has no weight
    return f / len(label_set)


def f1_score_2_value(real_labels: List[bool], predicted_labels: List[bool]) -> float:
    assert len(real_labels) == len(predicted_labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(real_labels)):
        if real_labels[i] and predicted_labels[i]:
            tp += 1
        elif not real_labels[i] and not predicted_labels[i]:
            tn += 1
        elif not real_labels[i] and predicted_labels[i]:
            fp += 1
        elif real_labels[i] and not predicted_labels[i]:
            fn += 1
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if recall + precision != 0:
        return 2 * (recall * precision) / (recall + precision)
    else:
        return 0


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    feature_array = np.array(features)
    result = np.empty((feature_array.shape[0], 0))
    for i in range(1, k + 1):
        result = np.concatenate((result, np.power(feature_array, i)), axis=1)
    return result.tolist()


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    a = np.array(point1)
    b = np.array(point2)
    return np.linalg.norm(a - b)


# TODO: clearify the meaning of inner product. 0 to any point is 0???
# TODO: [10,10] is nearer [1,1] than [9,9]?
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    a = np.array(point1, dtype=float)
    b = np.array(point2, dtype=float)
    inner_product = np.inner(a, b)
    return inner_product.reshape((1, 1))


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    k = euclidean_distance(point1, point2) ** 2
    k = k * -0.5
    return -np.exp(k)


def normalize(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[3, 4], [1, -1], [0, 0]],
    the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
    """
    result = []
    for i in range(len(features)):
        vector = np.array(features[i])
        norm = np.linalg.norm(vector)
        if norm == 0:
            result.append(vector.tolist())
        else:
            result.append((vector / norm).tolist())
    return result


def min_max_scale(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    """
    # TODO: confirm 0-1 normalization
    matrix = np.array(features, dtype=float)
    columns = list(range(0, matrix.shape[1]))  # column size
    numerator = matrix[:, columns] - matrix[:, columns].min(axis=0)
    denominator = (matrix[:, columns].max(axis=0) -
                   matrix[:, columns].min(axis=0))
    matrix[:, columns] = numerator / denominator

    return matrix.tolist()


class NormalizationScaler:
    def __init__(self):
        self.norm = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        # store norm
        result = []
        for i in range(len(features)):
            vector = np.array(features[i])
            if not isinstance(self.norm, np.ndarray):
                self.norm = np.linalg.norm(vector)
            if self.norm == 0:
                result.append(vector.tolist())
            else:
                result.append((vector / self.norm).tolist())
        return result


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min = None
        self.max = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        # wrong. do it for every column
        matrix = np.array(features, dtype=float)
        columns = list(range(0, matrix.shape[1]))  # column size
        if not isinstance(self.max, np.ndarray):
            self.max = np.amax(matrix, axis=0)
        if not isinstance(self.min, np.ndarray):
            self.min = np.amin(matrix, axis=0)
        numerator = matrix - self.min
        denominator = (self.max -
                       self.min)
        matrix[:, columns] = numerator / denominator
        return matrix.tolist()
