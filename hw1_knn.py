from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function
        self.training_data = None

    def train(self, features: List[List[float]], labels: List[int]):
        assert len(features) == len(labels)
        train_matrix = numpy.array(features)
        self.training_data = numpy.append(train_matrix, numpy.array(labels).reshape((-1, 1)), axis=1)

    def predict(self, features: List[List[float]]) -> List[int]:
        result = []
        for feature in features:
            neighbors = self.get_neighbors(numpy.array(feature).reshape((1, -1)))
            the_class = self.get_response(neighbors)
            result.append(the_class)
        return result

    def get_neighbors(self, instance):
        distances = []
        for x in self.training_data:
            dist = self.distance_function(instance, x[0: -1].reshape(1, -1))
            distances.append((x, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][0])
        return neighbors

    def get_response(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sorted_votes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
