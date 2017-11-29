import numpy as np
import matplotlib.pyplot as plt
from planar_data_classification.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_data_classification.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets



def layer_size(X, Y):
    # n_x:输入层size, n_h：隐藏层size ,n_y：输出层size
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
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

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    assert (A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logprobs1 = np.multiply(Y, np.log(A2))
    logprobs2 = np.multiply(1-Y, np.log(1-A2))
    cost = -np.sum(logprobs1 + logprobs2)/m

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m

    Z1 = cache["Z1"]
    f = np.tanh(Z1)
    f = 1 - f*f
    dZ1 = np.dot(W2.T, dZ2) * f
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)

    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        # 前向传播
        A2, cache = forward_propagation(X, parameters)
        #  计算损失
        cost = compute_cost(A2, Y, parameters)
        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 更新参数
        parameters = update_parameters(parameters, grads)
        # 输出
        if print_cost:
            print("Cost after iteration %i: %f"%(i, cost))

    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    i = 0
    for item in A2[0]:
        if item>0.5:
            A2[0][i]=1
        else:
            A2[0][i]=0
        i+=1
    predictions = A2
    return predictions



# X_assess, Y_assess = layer_sizes_test_case()
# n_x, n_h, n_y = layer_size(X_assess, Y_assess)
# print('The size of the input layer is :n_x=' + str(n_x))
# print('The size of the hidden layer is :n_h=' + str(n_h))
# print('The size of the output layer is :n_y=' + str(n_y))

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W1 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))

# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost=" + str(compute_cost(A2, Y_assess, parameters)))

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print(grads["dW1"])
# print(grads["db1"])
# print(grads["dW2"])
# print(grads["db2"])

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
# print(parameters["W1"])
# print(parameters["b1"])
# print(parameters["W2"])
# print(parameters["b2"])

# x_assess, y_assess = nn_model_test_case()
# parameters = nn_model(x_assess, y_assess, 4, num_iterations=10000, print_cost=True)
# print("W1:" + str(parameters["W1"]))
# print("b1:" + str(parameters["b1"]))
# print("W2:" + str(parameters["W2"]))
# print("b2:" + str(parameters["b2"]))

# parameters, X_assess = predict_test_case()
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))

X, Y = load_planar_dataset()
# parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title('Decision Boundary for hidden layer size ' + str(4))
# predictions = predict(parameters, X)
# # 预测正确的值在相乘的时候为1,预测错误的相乘后为0, 这里之所以要相加一下，是为了凑够整个样本数量;然后再除以中整个样本数量得到百分比。
# print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')
# plt.show()

# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y,n_h, num_iterations=5000)
#     plot_decision_boundary(lambda  x: predict(parameters, x.T), X, Y)
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
#     print('Accuracy for {} hidden units: {} %'.format(n_h, accuracy))
#     # plt.show()

#  尝试其它数据集
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "noisy_moons"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blibs":
    Y = Y%2

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()