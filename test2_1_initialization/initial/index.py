import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from initial.init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from initial.init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

# set display defaults
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # 显示图像的最大范围
plt.rcParams['image.interpolation'] = 'nearest'  # 差值方式
plt.rcParams['image.cmap'] = 'gray'              # 灰度空间

train_x, train_Y, test_x, test_Y = load_dataset()

def model(X, Y, learning_rate = 0.01, num_iteration = 15000, print_cost = True, initialization = 'he'):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dim = [X.shape[0], 10, 5, 1]

    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dim)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dim)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dim)

    for i in range(0, num_iteration):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters


def initialize_parameters_zeros(layers_dim):
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dim[l-1], layers_dim[l])).T
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return parameters

def initialize_parameters_random(layers_dim):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1])*10
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return parameters

def initialize_parameters_he(layers_dim):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * np.sqrt(2./layers_dim[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dim[l], 1))

    return parameters

# parameters = initialize_parameters_random([3, 2, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# parameters = model(train_x, train_Y, initialization='zeros')
# parameters = model(train_x, train_Y, initialization='random')
# print('On the train set:')
# prediction_train = predict(train_x, train_Y, parameters)
# print('On the test set:')
# prediction_test = predict(test_x, test_Y, parameters)
# plt.title("Model With large random initializatioin")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_Y)

# parameters = initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

parameters = model(train_x, train_Y, initialization='he')
print('On the train set:')
prediction_train = predict(train_x, train_Y, parameters)
print('On the test set:')
prediction_test = predict(test_x, test_Y, parameters)
plt.title("Model With large he initializatioin")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_Y)
