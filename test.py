def readMockData():
    import json
    f = open('testdata.txt', 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        data.append(json.loads(line))
        pass
    return data

data = readMockData()

import numpy
import scipy
if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)


from hw1_lr import LinearRegression, LinearRegressionWithL2Loss
from utils import mean_squared_error

import matplotlib.pyplot as plt

from data import generate_data_part_1

features, values = generate_data_part_1()
features = data[0]
values = data[1]
model = LinearRegression(nb_features=1)
model.train(features, values)

mse = mean_squared_error(values, model.predict(features))
print('[part 1.2]\tmse: {mse:.5f}'.format(mse=mse))

plt.scatter([x[0] for x in features], values, label='origin');
plt.plot([x[0] for x in features], model.predict(features), label='predicted');
plt.legend()

from data import generate_data_part_2

features, values = generate_data_part_2()
features = data[2]
values = data[3]
model = LinearRegression(nb_features=1)
model.train(features, values)

mse = mean_squared_error(values, model.predict(features))
print('[part 1.3.1]\tmse: {mse:.5f}'.format(mse=mse))

plt.scatter([x[0] for x in features], values, label='origin');
plt.plot([x[0] for x in features], model.predict(features), label='predicted');
plt.legend()

from utils import polynomial_features

features, values = generate_data_part_2()
features = data[2]
values = data[3]

plt.scatter([x[0] for x in features], values, label='origin');

for k in [2, 4, 8]:
    features_extended = polynomial_features(features, k)
    #print(features_extended)
    model = LinearRegression(nb_features=k)
    model.train(features_extended, values)
    mse = mean_squared_error(values, model.predict(features_extended))
    print('[part 1.3.2]\tk: {k:d}\tmse: {mse:.5f}'.format(k=k, mse=mse))
    plt.plot([x[0] for x in features], model.predict(features_extended), label='k={k}'.format(k=k));
plt.legend()

from data import generate_data_part_3

features, values = generate_data_part_3()
features = data[4]
values = data[5]

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
    print('[part 1.4.1]\tk: {k:d}\t'.format(k=k) +
          'train mse: {train_mse:.5f}\tvalid mse: {valid_mse:.5f}'.format(
              train_mse=train_mse, valid_mse=valid_mse))

    if valid_mse < best_mse:
        best_mse, best_k = valid_mse, k

combined_features_extended = polynomial_features(train_features + test_features, best_k)
model = LinearRegression(nb_features=best_k)
model.train(combined_features_extended, train_values + test_values)

test_features_extended = polynomial_features(test_features, best_k)
test_mse = mean_squared_error(test_values, model.predict(test_features_extended))
print('[part 1.4.1 Linear Regression]\tbest_k: {best_k:d}\ttest mse: {test_mse:.5f}'.format(
    best_k=best_k, test_mse=test_mse))

from itertools import product

best_mse, best_k, best_alpha = 1e10, -1, -1
for k, alpha in product([1, 3, 10], [0.01, 0.1, 1, 10]):
    train_features_extended = polynomial_features(train_features, k)
    model = LinearRegressionWithL2Loss(nb_features=k, alpha=alpha)
    model.train(train_features_extended, train_values)
    train_mse = mean_squared_error(train_values, model.predict(train_features_extended))

    valid_features_extended = polynomial_features(valid_features, k)
    valid_mse = mean_squared_error(valid_values, model.predict(valid_features_extended))
    print('[part 1.4.2]\tk: {k:d}\talpha: {alpha}\t'.format(k=k, alpha=alpha) +
          'train mse: {train_mse:.5f}\tvalid mse: {valid_mse:.5f}'.format(
              train_mse=train_mse, valid_mse=valid_mse))

    if valid_mse < best_mse:
        best_mse, best_k, best_alpha = valid_mse, k, alpha

combined_features_extended = polynomial_features(train_features + test_features, best_k)
model = LinearRegressionWithL2Loss(nb_features=best_k, alpha=best_alpha)
model.train(combined_features_extended, train_values + test_values)

test_features_extended = polynomial_features(test_features, best_k)
test_mse = mean_squared_error(test_values, model.predict(test_features_extended))
print('[part 1.4.2]\tbest_k: {best_k:d}\tbest_alpha: {best_alpha:f}\t'.format(
    best_k=best_k, best_alpha=best_alpha) +
      'test mse: {test_mse:.5f}'.format(test_mse=test_mse))


from hw1_knn import KNN
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance
from utils import f1_score

distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
}

from data import generate_data_cancer
features, labels = generate_data_cancer()

features = data[6]
labels = data[7]

train_features, train_labels = features[:400], labels[:400]
valid_features, valid_labels = features[400:460], labels[400:460]
test_features, test_labels = features[460:], labels[460:]

assert len(train_features) == len(train_labels) == 400
assert len(valid_features) == len(valid_labels) == 60
assert len(test_features) == len(test_labels) == 109

for name, func in distance_funcs.items():
    best_f1_score, best_k = -1, 0
    for k in [1, 3, 10, 20, 50]:
        model = KNN(k=k, distance_function=func)
        model.train(train_features, train_labels)
        train_f1_score = f1_score(
            train_labels, model.predict(train_features))

        valid_f1_score = f1_score(
            valid_labels, model.predict(valid_features))
        print('[part 2.1] {name}\tk: {k:d}\t'.format(name=name, k=k) +
              'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
              'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

        if valid_f1_score > best_f1_score:
            best_f1_score, best_k = valid_f1_score, k

    model = KNN(k=best_k, distance_function=func)
    model.train(train_features + valid_features,
                train_labels + valid_labels)
    test_f1_score = f1_score(test_labels, model.predict(test_features))
    print()
    print('[part 2.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
          'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
    print()





from utils import NormalizationScaler, MinMaxScaler

scaling_functions = {
    'min_max_scale': MinMaxScaler,
    'normalize': NormalizationScaler,
}

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
            print(
                '[part 2.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
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

import matplotlib.pyplot as plt
from data import generate_data_perceptron
from hw1_perceptron import Perceptron
import numpy as np

## To clearly visualize the problem, we just use 2 features for now
## y = f(x1,x2)

nb_features=2
model = Perceptron(nb_features=nb_features)
x, y = generate_data_perceptron(nb_features=nb_features, seperation=2)
x = data[8]
y = data[9]
x1 = []
x2 = []
for i in x:
    x1.append(i[1])
    x2.append(i[2])
plt.scatter(x1, x2, c=y)
##plt.show()

converged = model.train(x, y)
y_hat = model.predict(x)
correct = 0
for i, y_real in enumerate(y):
    if (y_hat[i] == y_real):
        correct = correct + 1

print("Accuracy on training data is {}".format(correct * 100 / len(y)))
print(correct)

w = model.get_weights()
x1 = []
x2 = []
for i in x:
    x1.append(i[1])
    x2.append(i[2])
plt.scatter(x1, x2, c=y)
plt.plot(np.arange(-3, 3, 1), -(w[1] * np.arange(-3, 3, 1) + w[0]) / w[2])
##plt.show()

model.reset()
print(model.get_weights())
test_x, train_x = x[80:], x[:80]
test_y, train_y = y[80:], y[:80]
converged = model.train(train_x, train_y)
y_hat = model.predict(test_x)
correct = 0
for i, y_real in enumerate(test_y):
    if (y_hat[i] == y_real):
        correct = correct + 1

print("Accuracy on testing data is {}".format(correct * 100 / len(test_y)))

w = model.get_weights()
x1 = []
x2 = []
for i in x:
    x1.append(i[1])
    x2.append(i[2])
plt.scatter(x1, x2, c=y)
plt.plot(np.arange(-3, 3, 1), -(w[1] * np.arange(-3, 3, 1) + w[0]) / w[2])
#plt.show()

nb_features=2
model = Perceptron(nb_features=nb_features)
x, y = generate_data_perceptron(nb_features=nb_features, seperation=1)
# plot and satisfy your self that data is not linearly seperable
x = data[10]
y = data[11]
x1 = []
x2 = []
for i in x:
    x1.append(i[1])
    x2.append(i[2])
plt.scatter(x1, x2, c=y)
#plt.show()

converged = model.train(x, y)
if (converged):
    print('Algorithm has converged')
else:
    print('Algorithm didnot converge')

y_hat = model.predict(x)
correct = 0
for i, y_real in enumerate(y):
    if (y_hat[i] == y_real):
        correct = correct + 1

print("Accuracy on training data is {}".format(correct * 100 / len(y)))
print(correct)

w = model.get_weights()
x1 = []
x2 = []
for i in x:
    x1.append(i[1])
    x2.append(i[2])
plt.scatter(x1, x2, c=y)
plt.plot(np.arange(-3, 3, 1), -(w[1] * np.arange(-3, 3, 1) + w[0]) / w[2])
#plt.show()

nb_features=10
model = Perceptron(nb_features=nb_features)

# use seperation=1 for non-seperable
# use seperation=2 for seperable

x, y = generate_data_perceptron(nb_features=nb_features, seperation=1)
# plot first two dimensions

x = data[12]
y = data[13]

x1 = []
x2 = []
for i in x:
    x1.append(i[1])
    x2.append(i[2])
plt.scatter(x1, x2, c=y)
#plt.show()

converged = model.train(x, y)
if (converged):
    print('Algorithm has converged')
else:
    print('Algorithm didnot converge')

y_hat = model.predict(x)
correct = 0
for i, y_real in enumerate(y):
    if (y_hat[i] == y_real):
        correct = correct + 1

print("Accuracy on training data is {}".format(correct * 100 / len(y)))
print(correct)