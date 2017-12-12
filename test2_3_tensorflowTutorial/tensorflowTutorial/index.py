import math
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflowTutorial.tf_utils import load_dataset, random_mini_batches,  convert_to_one_hot, predict

np.random.seed(1)


# y_hat = tf.constant(36, name='y_hat')
# y = tf.constant(39, name='y')
# loss = tf.Variable((y - y_hat)**2, name='loss')
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(loss))

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)
# print(c)
# sess = tf.Session()
# print(sess.run(c))

# sess = tf.Session()
# x = tf.placeholder(tf.int64, name='x')
# print(sess.run(2*x, feed_dict={x: 3}))
# sess.close()

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.constant(np.random.randn(4,3), name='W')
    b = tf.constant(np.random.randn(4,1), name='b')
    result = tf.Variable(tf.add(tf.matmul(W, X), b))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    result = sess.run(result)

    sess.close()
    return result

# print('result = ' + str(linear_function()))

def sigmoid(Z):
    x = tf.placeholder(tf.float32, name='x')
    res = tf.sigmoid(x)
    with tf.Session() as sess:
        res = sess.run(res, feed_dict={x:Z})
        sess.close()

    return res

# print('sigmoid(0) = ' + str(sigmoid(0)))
# print('sigmoid(12) = ' + str(sigmoid(12)))

def cost(logits, labels):
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')
    init = tf.global_variables_initializer()
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    with tf.Session() as sess:
        sess.run(init)
        cost = sess.run(cost, feed_dict={z: logits, y: labels})
        sess.close()

    return cost


# logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
# cost = cost(logits, np.array([0, 0, 1, 1]))
# print('Cost = ' + str(cost))


def one_hot_matrix(labels, C):
    one_hot = tf.one_hot(labels, C, axis=0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot)
        sess.close()

    return one_hot

# labels = np.array([1, 2, 3, 0, 2, 1])
# one_hot = one_hot_matrix(labels, C = 4)
# print('one_hot = ' + str(one_hot))

def ones(shape):
    ones = tf.ones(shape)
    with tf.Session() as sess:
        ones = sess.run(ones)
        sess.close()
    return ones

def zeros(shape):
    zeros = tf.zeros(shape)
    with tf.Session() as sess:
        zeros = sess.run(zeros)
        sess.close()
    return zeros

# print('ones = ' + str(ones([5, 4])))
# print('zeros =  ' + str(zeros((5,4))))


X_train_org, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# print(X_train_org.shape)
# print(Y_train_orig.shape)
# print(X_test_orig.shape)

# index = 0
# plt.imshow(X_train_org[index])
# print('y = ' + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()

X_train_flatten = X_train_org.reshape(X_train_org.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
# print('number of train examples = ' + str(X_train.shape[1]))
# print('number of test examples = ' + str(X_test.shape[1]))
# print('X_train shape:' + str(X_train.shape))
# print('Y_train shape: ' + str(Y_train.shape))
# print('X_test shape: ' + str(X_test.shape))
# print('Y_test shape: ' + str(Y_test.shape))

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))

    return X, Y

# X, Y = create_placeholders(12288, 6)
# print('X = ' + str(X))
# print('Y = ' + str(Y))

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

    return parameters

# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print('W1 = ' + str(parameters['W1']))
#     print('b1 = ' + str(parameters['b1']))
#     print('W2 = ' + str(parameters['W2']))
#     print('b2 = ' + str(parameters['b2']))
#     print('W3 = ' + str(parameters['W3']))
#     print('b3 = ' + str(parameters['b3']))
#     sess.close()

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print('Z3 = ' + str(Z3))


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    Y = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print('Cost = ' + str(cost))

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_x, Y: minibatch_y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost and epoch % 100 ==0:
                print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration (per tens)')
        plt.title('Learning_rate = ' + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print('Parameters have been trained!')

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('Train accuray: ', accuracy.eval({X: X_train, Y: Y_train}))
        print('Test accuray: ', accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)

# import scipy
# from PIL import Image
# from scipy import ndimage
# my_image = '3_orig.jpg'
# my_image = '3.jpg'
# fname = 'images/' + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
# my_image_prediction = predict(my_image, parameters)

# plt.imshow(image)
# print('y = ' + str(np.squeeze(my_image_prediction)))

