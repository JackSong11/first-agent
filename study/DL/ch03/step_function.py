# coding: utf-8
import numpy as np
import matplotlib

matplotlib.use('MacOSX')  # 或者 'Qt5Agg'，取决于你系统安装了哪个
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=int)


X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 図で描画するy軸の範囲を指定
plt.show()