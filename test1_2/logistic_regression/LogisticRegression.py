import numpy as np
from logistic_regression.lr_utils import load_dataset


class LogisticRegression(object):
    def __init__(self):
        pass

    def sigmoid(self, z):
        res = 1./(1 + np.exp(-z))
        return res



    def initialize_with_zeros(self, dim):
        # w = np.zeros((dim, 1), float)
        w = np.multiply(100, np.random.randn(dim, 1))
        b = 0.
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b

    def propagate(self, w, b, X, Y):
        m = X.shape[1]
        Z = np.dot(w.T, X) + b
        A = self.sigmoid(Z)
        # a = np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A))
        # cost = -np.sum(a, axis=1)/m
        cost = -np.sum(np.multiply(Y, np.log(A) + np.multiply(1-Y, np.log(1-A))), axis=1)/m
        #cost1 = float(cost*100000)
        print("cost:%f" % cost)
        dw = (np.dot(X, (A-Y).T))/m
        db = np.sum(A-Y, axis=1)/m
        grads = {"dw": dw, "db": db}

        return cost, grads

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        costs=[]
        for i in range(num_iterations):
            cost, grads = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate*dw
            b = b - learning_rate*db
            if i%100 ==0:
                costs.append(cost)

            if print_cost and i%100 ==0:
                print("Cost after iteration %i: %.5f"%(i, cost))

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        return params, grads,  costs

    def predict(self, w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        Z = np.dot(w.T, X) + b
        A = self.sigmoid(Z)
        for i in range(A.shape[1]):
            if(A[0, i]>0.5):
                Y_prediction[0, i] =1
            elif(A[0, i]<0.5):
                Y_prediction[0, i] = 0
        assert(Y_prediction.shape ==(1, m))
        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=200, learning_rate=0.5, print_cost=False):
        w, b = self.initialize_with_zeros(X_train.shape[0])
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)
        print("train accuracy:{} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
        print("test accuracy:{} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        d = {"costs": costs,
             "y_predictjion_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d

    pass

lg = LogisticRegression()
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
# cost, grads = lg.propagate(w, b, X, Y)
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))
# print("cost = " + str(cost))
# print("\n*****************")
# params, grads, costs = lg.optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
# print("w=" + str(params["w"]))
# print("b=" + str(params["b"]))
# print("dw=" + str(grads["dw"]))
# print("db=" + str(grads["db"]))
# print("\n*****************")
# print("prediction = " + str(lg.predict(w, b, X)))
# print("\n*****************")



X_train, Y_train, X_test, Y_test, _ = load_dataset()
X_train = X_train.reshape(209, 64*64*3).T
X_test = X_test.reshape(50, 64*64*3).T
d = lg.model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005, print_cost=True)

