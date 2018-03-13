from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''

        self.nb_features = nb_features
        self.w = [margin for i in range(0, nb_features + 1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        converge = False
        w_vector = np.array(self.w)
        for k in range(self.max_iteration):
            update = False
            for i in range(len(features)):
                xn = np.array(features[i]).reshape((1, self.nb_features + 1))
                yn = labels[i]
                y = self.predict_one_value(w_vector, xn)
                if y is not yn:
                    w_vector = w_vector + xn.dot((yn - y)) / np.linalg.norm(xn)
                    update = True
            if not update:
                converge = True
                break  # converge
        self.w = w_vector.flatten().tolist()
        return converge

    def reset(self):
        self.w = [0 for i in range(0, self.nb_features + 1)]

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        results = []
        weights = np.array(self.w)
        for x in features:
            results.append(self.predict_one_value(weights, np.array(x).reshape((1, self.nb_features + 1))))
        return results

    def predict_one_value(self, weights, feature):
        result = weights.reshape(1, self.nb_features + 1).dot(feature.transpose())
        if result >= self.margin:
            return 1
        else:
            return -1

    def get_weights(self) -> List[float]:
        return self.w
