import numpy as np
import matplotlib.pyplot as plt
from planar_data_classification.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_data_classification.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets



np.random.seed(1)

X, Y = load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
# plt.xlabel('XXX横坐标')
# plt.ylabel('YYY纵坐标')
# # plt.legend(prop='unicode')  用来纠正中文编码问题
# plt.show();

shape_X = X.shape
shape_Y = Y.shape
num_samples = X.shape[1]
print("num_samples:{}".format(num_samples))
print("the shape of X: {}".format(shape_X))
print("the shape of Y: {}".format(shape_Y))


clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
# plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
# Print accuracy
LR_predictions = clf.predict(X.T)
plt.show()
print('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) \
                                                     + '%' + '(percentage of correctly labelled datapoints)')

