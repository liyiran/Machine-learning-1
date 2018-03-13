from unittest import TestCase
from hw1_perceptron import Perceptron
from data import generate_data_perceptron
import numpy as np
import matplotlib.pyplot as plt


class TestPerceptron(TestCase):
    def test_train(self):
        # def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        x = [[1, 63], [1, 64], [1, 65], [1, 66], [1, 67], [1, 68], [1, 69], [1, 71], [1, 71], [1, 72]]
        y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        perceptron = Perceptron(nb_features=1, max_iteration=100000, margin=1e-4)
        converge = perceptron.train(x, y)
        print(perceptron.get_weights())
        self.assertTrue(converge)

    def test_big_data_set(self):
        dataset = [[1, 2.7810836, 2.550537003],
                   [1, 1.465489372, 2.362125076],
                   [1, 3.396561688, 4.400293529],
                   [1, 1.38807019, 1.850220317],
                   [1, 3.06407232, 3.005305973],
                   [1, 7.627531214, 2.759262235],
                   [1, 5.332441248, 2.088626775],
                   [1, 6.922596716, 1.77106367],
                   [1, 8.675418651, -0.242068655],
                   [1, 7.673756466, 3.508563011]]
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        perceptron = Perceptron(nb_features=2, max_iteration=10, margin=1e-4)
        perceptron.train(dataset, y)
        print(perceptron.get_weights())
        print(perceptron.predict(dataset))

    def test_integration1(self):
        nb_features = 2
        model = Perceptron(nb_features=nb_features, max_iteration=10)
        x, y = generate_data_perceptron(nb_features=nb_features, seperation=2)
        x1 = []
        x2 = []
        for i in x:
            x1.append(i[2])
            x2.append(i[1])
        plt.scatter(x1, x2, c=y)
        plt.show()
        converged = model.train(x, y)
        self.assertTrue(converged)
        y_hat = model.predict(x)
        correct = 0
        for i, y_real in enumerate(y):
            if (y_hat[i] == y_real):
                correct = correct + 1
            else:
                print(y_hat[i], y_real)
        print("Accuracy on training data is {}".format(correct * 100 / len(y)))
        print(correct)
        w = model.get_weights()
        x1 = []
        x2 = []
        for i in x:
            x1.append(i[1])
            x2.append(i[2])
        plt.scatter(x1, x2, c=y)
        plt.plot(np.arange(-3, 3, 1), -(w[1] * np.arange(-3, 3, 1) + w[0]) / w
        [2])
        plt.show()
        model.reset()
        print (model.get_weights())
        test_x, train_x = x[80:], x[:80]
        test_y, train_y = y[80:], y[:80] 
        converged = model.train(train_x, train_y) 
        self.assertTrue(converged)
        y_hat = model.predict(test_x)
        correct = 0
        for i, y_real in enumerate(test_y):
            if (y_hat[i]==y_real): correct = correct + 1
        print ("Accuracy on testing data is {}".format(correct*100/len(test_y) ))
        w = model.get_weights() 
        x1 = []
        x2 = []
        for i in x:
            x1.append(i[1])
            x2.append(i[2])
        plt.scatter(x1, x2, c=y)
        plt.plot(np.arange(-3, 3, 1), -(w[1] * np.arange(-3, 3, 1) + w[0]) / w
        [2])
        plt.show()
    
    def test_not_seperate(self):
        nb_features=2
        model = Perceptron(nb_features=nb_features)
        x, y = generate_data_perceptron(nb_features=nb_features, seperation=1) # plot and satisfy your self that data is not linearly seperable
        x1 = []
        x2 = []
        for i in x:
            x1.append(i[2])
            x2.append(i[1])
        plt.scatter(x1, x2, c=y)
        plt.show()
        converged = model.train(x, y) 
        if (converged):
            print ('Algorithm has converged') 
        else:
            print ('Algorithm didnot converge')
        y_hat = model.predict(x) 
        correct = 0
        for i, y_real in enumerate(y):
            if (y_hat[i]==y_real): 
                correct = correct + 1
        print ("Accuracy on training data is {}".format(correct*100/len(y))) 
        print (correct)
        w = model.get_weights() 
        x1 = []
        x2 = []
        for i in x:
            x1.append(i[1])
            x2.append(i[2])
        plt.scatter(x1, x2, c=y)
        plt.plot(np.arange(-3, 3, 1), -(w[1] * np.arange(-3, 3, 1) + w[0]) / w[2])
        plt.show()

    def test_high_d(self):
        nb_features=10
        model = Perceptron(nb_features=nb_features)
        # use seperation=1 for non-seperable
        # use seperation=2 for seperable
        x, y = generate_data_perceptron(nb_features=nb_features, seperation=1)
        # plot first two dimensions
        x1 = []
        x2 = []
        for i in x:
            x1.append(i[2])
            x2.append(i[1])
        plt.scatter(x1, x2, c=y)
        plt.show()
        converged = model.train(x, y) 
        if (converged):
            print ('Algorithm has converged') 
        else:
            print ('Algorithm didnot converge')
        y_hat = model.predict(x) 
        correct = 0
        for i, y_real in enumerate(y):
            if (y_hat[i]==y_real): correct = correct + 1
        print ("Accuracy on training data is {}".format(correct*100/len(y))) 
        print (correct)
 