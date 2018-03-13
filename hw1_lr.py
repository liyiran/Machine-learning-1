from __future__ import division, print_function

from typing import List

import numpy
import scipy
import scipy.linalg


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.omega = None

    def train(self, features: List[List[float]], values: List[float]):
        features_matrix = numpy.array(features)
        big_x = numpy.insert(features_matrix, 0, 1, axis=1)
        y = numpy.empty((0, 1), dtype=float)
        for i in values:
            y = numpy.append(y, numpy.array(i))
        y = y.reshape((-1, 1))
        # TODO: confirm the number of features
        omega = numpy.linalg.solve(big_x.transpose().dot(big_x), big_x.transpose().dot(y))
        self.omega = omega

    def predict(self, features: List[List[float]]) -> List[float]:
        if self.omega is None:
            raise ValueError("invoke train() before predict")
        else:
            y = numpy.insert(numpy.array(features).transpose(), 0, 1, axis=0)
            return self.omega.transpose().dot(y).flatten().tolist()

    def get_weights(self) -> List[float]:
        if self.omega is None:
            return []
        else:
            return self.omega.transpose().flatten().tolist()


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''

    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        features_matrix = numpy.array(features)
        big_x = numpy.insert(features_matrix, 0, 1, axis=1)
        y = numpy.empty((0, 1), dtype=float)
        for i in values:
            y = numpy.append(y, numpy.array(i))
        y = y.reshape((-1, 1))
        omega = numpy.linalg.solve(big_x.transpose().dot(big_x) + self.alpha * numpy.identity(big_x.shape[1], dtype=float), big_x.transpose().dot(y))
        self.omega = omega

    def predict(self, features: List[List[float]]) -> List[float]:
        if self.omega is None:
            raise ValueError("invoke train() before predict")
        else:
            y = numpy.insert(numpy.array(features).transpose(), 0, 1, axis=0)
            return self.omega.transpose().dot(y).flatten().tolist()

    def get_weights(self) -> List[float]:
        if self.omega is None:
            return []
        else:
            return self.omega.transpose().flatten().tolist()


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
