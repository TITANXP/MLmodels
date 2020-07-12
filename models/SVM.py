import random
import numpy as np
import matplotlib.pyplot as plt

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
            y.append(-1)
    y = np.array(y)
    return X, y

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

def selectJrand(i, m):
    """
    在范围内随机选择一个整数
    :param i: alpha的下标
    :param m: 所有alpha的数目
    :return:
    """
    j = i;
    while(j == i):
        j = int(random.randint(0, m-1))
    return j

def clipAlpha(aj, H, L):
    """数值太大或太小时做出限制"""
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smo_simple(X, y, C, toler, max_iter):
    """
    简化版SMO算法
    :param X:数据集
    :param y: 类别标签
    :param C: 常数
    :param toler: 容错率
    :param max_iter: 退出前最大循环次数
    :return:
    """
    X = np.mat(X)
    y = np.mat(y).transpose()
    b = 0
    m, n = np.shape(X)
    alphas = np.zeros((m, 1))
    iter = 0
    while(iter < max_iter):
        alpha_pairs_changed = 0 # 记录alpha是否已被优化
        for i in range(m):
            fXi = float(np.multiply(alphas, y).T * (X * X[i,:].T)) + b
            Ei = fXi - float(y[i])
            if ((y[i] * Ei < -toler) and (alphas[i] < C)) or ((y[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选取第二个alpha值
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, y).T * (X * X[j,:].T)) + b
                Ej = fXj - float(y[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
            # 保证alpha在0——C之间
            if y[i] != y[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, alphas[j] - alphas[i])
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                print('L==H')
                continue

            # eta = k11 + k22 - 2k12
            eta = 2.0 * X[i,:] * X[j,:].T - X[i,:] * X[i,:].T - X[j,:] * X[j,:].T
            if eta > 0:
                print("eta > 0")
                continue
            alphas[j] -= y[i] * (Ei - Ej) / eta
            alphas[j] = clipAlpha(alphas[j], H, L)
            if abs(alphas[j] - alpha_j_old) < 0.00001:
                print('j not moving enough')
                continue

            # 更新alpha1，从而满足限制条件
            alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

            b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * X[i,:]*X[i,:].T - y[j] * (alphas[j] - alpha_j_old) * X[i,:]*X[j,:].T
            b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * X[i,:]*X[j,:].T - y[j] * (alphas[j] - alpha_j_old) * X[j,:]*X[j,:].T

            if alphas[i] > 0 and alphas[i] < C:
                b = b1
            elif alphas[j] > 0 and alphas[j] < C:
                b = b2
            else:
                b = (b1 + b2) / 2.0
            alpha_pairs_changed += 1
            print("iter: %d i: %d pair changed %d" % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 1:
            iter =+ 1
        else:
            iter = 0
        print("iter num: %d" % iter)
    return b, alphas

if __name__ == '__main__':
    X, y = load_data()
    b, alphas = smo_simple(X, y, 0.6, 0.01, 40)



