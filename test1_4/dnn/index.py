import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn.testCases_v2 import *
from dnn.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize']=(5.0 ,4.0) # set  default size of plts
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def initialize_parameters_deep(layer_dims):
    # layer_dims -------python array(list) containing the dimensions of each layer in our network
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], (layer_dims[l-1]))*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, _= sigmoid(Z)
        activation_cache = Z     # 这里存储cache的时候没必要把w, b也放进去吧？目前就存储一个Z。
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, _ = relu(Z)
        activation_cache = Z  # 要不要只存储一个Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL))
    cost = -np.sum(cost, axis=1, keepdims=True)/m
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost

def linear_backword(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backword(dA, cache, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache

    # 下面对应每一种方法，要计算相应的导数（令sigmoid函数为f(x),则f'(x) = f(x)*f(x); 然而relu函数的导数为分段函数，要分别计算）
    if activation == 'relu':
        f = Z
        for i in range(0, f.shape[0]):
            for j in range(0, f.shape[1]):
                if f[i][j] > 0:
                    f[i][j] = 1
                elif f[i][j] < 0:
                    f[i][j] = 0
        dZ = dA * f
        dA_prev, dW, db = linear_backword(dZ, linear_cache)
    elif activation == 'sigmoid':
        fz, _ = sigmoid(Z)
        dZ = dA * fz * (1-fz)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backword(dAL, current_cache, activation='sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backword(grads["dA" + str(l+2)], current_cache, activation='relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# parameters = initialize_parameters(2, 2, 1)
# print(parameters["W1"])
# print(parameters["b1"])
# print(parameters["W2"])
# print(parameters["b2"])

# parameters = initialize_parameters_deep([100, 100, 50, 50, 50, 10, 1])
# print(parameters["W1"])
# print(parameters["b1"])
# print(parameters["W2"])
# print(parameters["b2"])
# print(parameters["W3"])
# print(parameters["b3"])
# print(parameters["W4"])
# print(parameters["b4"])
# print(parameters["W5"])
# print(parameters["b5"])
# print(parameters["W6"])
# print(parameters["b6"])

# A, W, b = linear_forward_test_case()
# Z, linear_cache = linear_forward(A, W, b)
# print('Z = ' + str(Z))

# A_prev, W, b = linear_activation_forward_test_case()
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation='sigmoid')
# print('With sigmoid: A = ' + str(A))
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation='relu')
# print('With relu: A = ' + str(A))

# X, parameters = L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))

# dZ, linear_cache = linear_backward_test_case()
# dA_prev, dW, db = linear_backword(dZ, linear_cache)
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db))

# AL, linear_activation_cache = linear_activation_backward_test_case()
# dA_prev, dW, db = linear_activation_backword(AL, linear_activation_cache, activation='sigmoid')
# print('sigmoid:')
# print('dA_prev = ' + str(dA_prev))
# print('dW = ' + str(dW))
# print('db = ' + str(db))
# dA_prev, dW, db = linear_activation_backword(AL, linear_activation_cache, activation='relu')
# print('relu:')
# print('dA_prev = ' + str(dA_prev))
# print('dW = ' + str(dW))
# print('db = ' + str(db))

# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print("dW1  :\n" + str(grads["dW1"]))
# print("db1  :\n" + str(grads["db1"]))
# print("dA1  :\n" + str(grads["dA1"]))

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))