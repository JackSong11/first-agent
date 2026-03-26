# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('MacOSX') # 或者 'Qt5Agg'，取决于你系统安装了哪个
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()