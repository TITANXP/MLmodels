'''
推荐系统 隐语义模型
'''
import numpy as np

def load_data():
    return [
        [4, 4, 0, 2, 2],
        [4, 0, 0, 3, 3],
        [4, 0, 0, 1, 1],
        [1, 1, 1, 2, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0]
    ]

def LFM_grad_desc(R, epoch, lambda_, K=3, alpha=0.0001):
    """
    R = P * Q
    梯度下降求P和Q
    :param R:
    :param epoch:
    :param lambda_: 正则化系数
    :param K: 隐特征向量维度
    :param alpha:
    :return: 分解后的P Q
            P：初始化用户特征矩阵 MxK
            Q：初始化物品特征矩阵 NxK
    """
    R = np.mat(R)
    M,N = R.shape

    # 随机初始化 P Q
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T

    for e in range(epoch):
        # 对所有的用户u，物品i遍历，对应的特征向量Pu、Qi梯度下降
        for u in range(M):
            for i in range(N):
                # 对每个进行过评分的求误差
                if R[u, i] > 0:
                    eui = np.dot(P[u,:], Q[:,i]) - R[u,i]
                    # 代入公式，更新当前的Pu，Qi
                    P[u] = P[u] - alpha * (2 * eui * Q[:,i] + 2 * lambda_ * P[u])
                    Q[:,i] = Q[:,i] - alpha * (2 * eui * P[u] + 2 * lambda_ * Q[:,i])
        # 计算当前损失
        cost = 0
        cost = np.power(np.dot(P, Q) - R,2)[R != 0].sum()
        if cost < 0.0001:
            break
        print('epoch',epoch, '误差：',cost)
    return P, Q.T, cost

if __name__ == '__main__':
    K = 5
    epochs = 5000
    alpha = 0.0002
    lambda_ = 0.004
    R = load_data()
    P, Q, cost = LFM_grad_desc(R, epochs, lambda_, K, alpha)
    print(R)
    print('预测')
    print(np.dot(P, Q))