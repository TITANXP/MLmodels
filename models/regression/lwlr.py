"""        局部加权线性回归
   线性回归容易出现欠拟合的现象， 因为他求的是具有最小均方误差的无偏估计
  所以需要在模型中引入一下偏差，从而降低预测的均方误差
      其中一个方法是局部加权线性回归：
          该算法给待预测点附近的每一个点赋予权重
"""
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

def lwlr(test_point, X, y, k=1.0):
    """
        局部加权线性回归
    给待预测点附近的每一个点赋予一个权重
    使用核来对附近的点赋予更高的权重，常用的是高斯核
    :param test_point: 需要进行预测的点
    :param X:
    :param y:
    :param k: 高斯核的参数，k越小，用于被用于训练模型的点越少
    :return:
    """
    X = np.mat(X)
    y = np.mat(y).T
    m = np.shape(X)[0]
    # m阶对角矩阵
    weights = np.mat(np.eye(m))
    for j in range(m):
        # 权重大小以指数级衰减, 与测试点距离越大权重越小
        diff = test_point - X[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    xTx = X.T * (weights * X)
    if np.linalg.det(xTx) == 0:
        print("矩阵不可逆")
        return
    W = xTx.I * (X.T * (weights * y))
    return test_point*W

def lwlr_test(X_test, X, y, k=1.0):
    m = np.shape(X_test)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(X_test[i], X, y, k)
    return y_hat


if __name__ == '__main__':
    k = 1
    # 加载数据
    X, y = load_data()
    print("局部加权线性回归")
    X_train = X[:100]; y_train = y[:100]
    X_test = X[-100:]; y_test = y[-100:]
    # 预测
    y_hat = lwlr_test(X_test, X_train, y_train, k)
    # 评估
    error = np.mean(np.abs(y_hat - y_test))
    print(y_hat)
    print(error)
    # 升序
    sorted_index = X_test[:,1].argsort(0)
    X_test_sorted = X_test[sorted_index][:,1]
    # 画图
    fig = plt.figure()
    plt.plot(X_test_sorted, y_test[sorted_index], c='red')
    plt.scatter(X_test[:,1],y_test, s=20)
    plt.title("k={}".format(k))
    plt.show()