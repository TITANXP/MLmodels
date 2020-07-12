import numpy as np
import matplotlib.pyplot as plt
"""
logistic 回归
梯度下降
"""

def load_data():
    weights = np.array([1, 0.6])
    # 随机初始化点
    X = np.random.randint(-50, 50, (500, 2))
    y = []
    # 在线上方的label为1，下方为0
    for x in X:
        result = np.dot(x, weights)
        if result >= 0:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    return X, y

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def grad_ascent(X, y):
    """ 梯度上升"""
    alpha = 0.01
    epoch = 300
    X = np.mat(X)
    y = np.mat(y).transpose()
    m,n = np.shape(X)
    weights = np.ones((n,1))
    for i in range(epoch):
         h = sigmoid(np.dot(X, weights))
         error = y - h
         weights = weights + alpha * X.transpose() * error
    return weights


def plot_line(X, y, weights):
    line_point_x = np.arange(-40, 40, 1)
    line_point_y = -weights[0] * line_point_x / weights[1]
    print(X[y==0][:,0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[y==0][:,0], X[y==0][:,1], s=30, c='red', marker='s')
    ax.scatter(X[y==1][:,0], X[y==1][:,1], s=30, c='blue')
    ax.plot(np.reshape(line_point_x, np.shape(line_point_y)), line_point_y)

    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    print(X)
    print(y)
    weights = grad_ascent(X, y)
    # 得到的weight与设定好的weight比例相同
    print(weights)
    plot_line(X,y, weights.getA())


