"""
前向逐步线性回归
    可以得到和lasso差不多的效果，但更加简单
    属于一种贪心算法，即每一步都尽可能减小误差
        一开始所有权重都设为1，然后每一步所作的决策是对某个权重增加或减小一个很小的值
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.scorer import mean_squared_error

def load_data():
    X = np.random.rand(200, 2)
    noise = np.random.randint(-1000, 1001, size=(200, )) / 10000
    # X[:,0] = 1
    # W0 相当于常数项b
    W = [2, 5]
    y = np.dot(X, W) + noise
    return X, y

def stage_wise(X, y, eps=0.01, epoch=200):
    X = np.mat(X); y = np.mat(y).T
    y_mean = y.mean()
    y = y - y_mean
    X_means = X.mean(0)
    X_vars = X.var(0)
    X = (X - X_means) / X_vars
    m, n = X.shape
    result_WS = np.zeros((epoch, n))
    WS = np.zeros((n,1))
    WS_test = WS.copy()
    WS_max = WS.copy()
    for i in range(epoch):
        print(WS.T)
        min_error = np.inf
        for j in range(n):
            for sign in [-1,1]:
                WS_test = WS.copy()
                WS_test[j] += eps * sign
                y_predict = X * WS_test
                error = mean_squared_error(y, y_predict)
                if error < min_error:
                    min_error = error
                    WS_max = WS_test
        WS = WS_max.copy()
        result_WS[i] = WS.T
    return result_WS

def plot_W(WS):
    plt.plot(WS)
    plt.legend(['w0', 'w1'])
    plt.xlabel('epoch')
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    WS = stage_wise(X, y)
    print(WS)
    plot_W(WS)
