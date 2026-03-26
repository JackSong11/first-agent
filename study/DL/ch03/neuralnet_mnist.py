# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from study.DL.dataset.mnist import load_mnist
from study.DL.common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

        # 层名称: b2, 数据形状: (100,)
        # 层名称: W1, 数据形状: (784, 50)
        # 层名称: b1, 数据形状: (50,)
        # 层名称: W2, 数据形状: (50, 100)
        # 层名称: W3, 数据形状: (100, 10)
        # 层名称: b3, 数据形状: (10,)

        # # 查看网络结构
        # for key, value in network.items():
        #     print(f"层名称: {key}, 数据形状: {value.shape}")
        #
        # # 如果想看具体数值（比如第一层的前 5 个权重）
        # print("\nW1 的部分数值:\n", network['W1'][:5, :5])
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 总共三层 第一层50个节点，第二层100个节点，第三层是输出层 10个节点
    a1 = np.dot(x, W1) + b1 # (1,50) = (1,784) (784,50) + (1,50)
    z1 = sigmoid(a1) # (1,50) = (1,50)
    a2 = np.dot(z1, W2) + b2 # (1,100) = (1,50) (50,100) + (1,100)
    z2 = sigmoid(a2) # (1,100) = (1,100)
    a3 = np.dot(z2, W3) + b3 # (1,10) = (1,100) (100,10) + (1,10)
    y = softmax(a3) # (1,10) = (1,10)

    return y


x, t = get_data()
# print(x.shape) # (10000, 784)
# print(t.shape) # (10000,)
# print(type(x)) # <class 'numpy.ndarray'>
# print(type(t)) # <class 'numpy.ndarray'>
# print(x) #
# print(t) # [7 2 1 ... 4 5 6]
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # np.argmax(y) 的作用是：找到数组中数值最大的那个元素的"索引"（下标）。
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))