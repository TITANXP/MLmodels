import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.random.rand(200, 2)
    noise = np.random.randint(-1000, 1001, size=(200, )) / 10000
    X[:,0] = 1
    # W0 相当于常数项b
    W = [2, 5]
    y = np.dot(X, W) + noise
    return X, y

def standRegres(X, y):
    X = np.mat(X)
    y = np.mat(y).T
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        # 行列式不为0可逆，反之不可逆
        print("矩阵不可逆")
        return
    # 最小二乘法
    W = xTx.I * (X.T * y)
    return W

def plot_line_and_point(X, y, W):
    # 画点
    plt.scatter(X[:,1], y, s=20)
    # 画线
    X_copy = X.copy()
    # 按升序排列， 如果直线上的数据点次序混乱，将会出现问题
    X_copy.sort(0)
    y_hat = np.dot(X_copy, W)
    plt.plot(X_copy[:, 1], y_hat, color='red')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    X, y = load_data()

    # 得到权重
    W = standRegres(X, y)
    print(W)
    # 进行预测
    y_hat = np.dot(X, W)
    # 画图
    plot_line_and_point(X, y, W)
    # 计算相关系数
    coef = np.corrcoef(y_hat.T, y)
    print(coef)


