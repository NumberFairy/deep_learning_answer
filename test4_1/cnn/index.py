import numpy as np
import h5py
import matplotlib.pyplot as plt

# set display  default
plt.rcParams['figure.figsize'] = (5.0, 4.0) # 显示图像的最大范围
plt.rcParams['image.interpolation'] = 'nearest' # 差值方式
plt.rcParams['image.cmap'] = 'gray' # 灰度空间


np.random.seed(1)

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    return X_pad

# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print('x.shape = ', x.shape)
# print('x_pad.shape = ', x_pad.shape)
# print('x[1,1] = ', x[1, 1])
# print('x_pad[1,1] =', x_pad[1, 1])
#
# fig, axarr = plt.subplots(1,2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_pad')
# axarr[1].imshow(x_pad[0,:,:,0])
# fig.show()
# print('hasdfadfhasdfa')


# 仅仅进行了一次卷积计算
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z

# np.random.seed(1)
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)
#
# Z = conv_single_step(a_slice_prev, W, b)
# print('Z = ' + str(Z))

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape #n_C_prev: the number of channles; n_C: the number of filter
    stride = hparameters['stride']
    pad = hparameters['pad']

    # compute the dimensions of the CONV output volume using the formula given above.
    n_H = int((n_H_prev + pad * 2 - f) / stride) + 1
    n_W = int((n_W_prev + pad * 2 - f) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c]) + b[:, :, :, c])
    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return Z, cache



# np.random.seed(1)
# A_prev = np.random.randn(10, 4, 4, 3)
# W = np.random.randn(2, 2, 3, 8)
# b = np.random.randn(1, 1, 1, 8)
# hparameters = {'stride': 1, 'pad': 2}
#
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean = ", np.mean(Z))
# print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])


def pool_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start: horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache

np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride": 1, "f": 4}

A, cache = pool_forward(A_prev, hparameters)
print('mode = max')
print('A = ', A)
print('*****')
A, cache = pool_forward(A_prev, hparameters, mode='avg')
print('mode = avg')
print('A = ', A)









































