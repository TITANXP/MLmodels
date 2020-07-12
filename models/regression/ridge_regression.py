"""
岭回归
    之前方法的问题：
        如果数据的特征比样本点还多，在计算(xtx)`-1时会出错，如果特征比样本点还多，说明X不是满秩矩阵，非满秩矩阵在求逆时会出问题
    为了解决这个问题，引入了岭回归
"""
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.random.rand(200, 2)
    noise = np.random.randint(-1000, 1001, size=(200, )) / 10000
    # X[:,0] = 1
    # W0 相当于常数项b
    W = [2, 5]
    y = np.dot(X, W) + noise
    return X, y

def ridge_regression(X, y, lambda_=0.2):
    """
    计算W
    岭回归：给XtX加上 λ*I，从而可以求逆
    :param X:
    :param y:
    :param lambda_:
    :return:
    """
    X = np.mat(X); y = np.mat(y).T
    m, n = np.shape(X)
    xTx = X.T * X
    denmo = xTx + lambda_ * np.eye(n)
    if np.linalg.det(denmo) == 0:
        print('矩阵行列式值为0， 不可逆')
        return
    W = denmo.I * (X.T * y)
    return W

def ridge_test(X, y):
    """
    得到30个不同的λ对应的W
    :param X:
    :param y:
    :return:
    """
    X = np.mat(X); y = np.mat(y)
    # 数据标准化
    y_mean = y.mean()
    y = y - y_mean
    X_means = X.mean(0)
    print(X_means)
    # 每个feature的方差
    X_var = X.var(0)
    X = (X - X_means) / X_var
    num_test_pts = 30
    WS = np.zeros((num_test_pts, X.shape[1]))
    for i in range(num_test_pts):
        W = ridge_regression(X, y, np.exp(i-10))
        WS[i,:] = W.T
    return WS

if __name__ == '__main__':
    X, y = load_data()
    WS = ridge_test(X, y)
    print(WS)
    # 画图
    # λ非常小时，W和普通回归一样，λ非常大时，所有W缩减为0
    plt.plot(WS)
    plt.legend(['w0','w1'])
    plt.xlabel('λ')
    plt.show()





