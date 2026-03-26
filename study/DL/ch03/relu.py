# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('MacOSX') # 或者 'Qt5Agg'，取决于你系统安装了哪个
import matplotlib.pylab as plt
# ... 其余代码不变

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()