import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from logistic_regression.lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


print(type(train_set_x_orig)), print(train_set_x_orig.shape)

print(type(train_set_y)), print(train_set_y.shape)

print(type(test_set_x_orig)), print(test_set_x_orig.shape)

print(type(test_set_y)), print(test_set_y.shape)


index = 0
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")
print("\n**************************\n")

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
print("train_set_x_flatten's type : " + str(type(train_set_x_flatten)))
print(train_set_x_flatten.shape)
print(train_set_x_flatten[0:5,0])
print(train_set_x_flatten)
print("\n*****************")
a = np.random.randn(5)
print(type(a))
print(a)
a = a.reshape(5,1)
print(a)

