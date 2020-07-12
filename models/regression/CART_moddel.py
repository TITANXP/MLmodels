"""
----模型树：每个叶子节点是一个线性方程
"""

import numpy as np
from sklearn.metrics.regression import mean_squared_error

def load_data():
    X = np.random.rand(200, 2)
    noise = np.random.randint(-1000, 1001, size=(200, )) / 10000
    W = [2, 5]
    y = np.dot(X, W) + noise
    dataSet = np.hstack((X, y.reshape(200,1)))
    return dataSet

def split_dataSet(dataSet, feature_index, feature_value):
    """
    使用二分法切分数据集
    :param dataSet:
    :param feature:待切分的特征
    :param feature_value:
    :return:
    """
    sub_dataSet0 = dataSet[dataSet[:,feature_index] > feature_value]
    sub_dataSet1 = dataSet[dataSet[:,feature_index] <= feature_value]
    return sub_dataSet0, sub_dataSet1

def linear_solve(dataSet):
    m, n = dataSet.shape
    X = np.mat(np.ones((m,n)))
    y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    y= np.mat(dataSet[:,-1]).T
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        print('不可逆')
        return
    W = xTx.I * (X.T * y)
    return W, X, y

def modelLeaf(dataSet):
    """创建叶节点: 一个线性方程"""
    W, X, y = linear_solve(dataSet)
    return W

def modelErr(dataSet):
    """计算label的总方差"""
    W, X, y = linear_solve(dataSet)
    y_hat = X * W
    return np.sum(np.power((y-y_hat), 2))

def choose_best_split(dataSet, leaf_type=modelLeaf, err_type=modelErr, ops=[1,4]):
    """

    :param dataSet:
    :param leaf_type:函数引用
    :param err_type: 函数引用
    :param ops:
    :return:
    """
    tolS = ops[0] # 容许的误差下降值
    tolN = ops[1] # 切分的最小样本数
    # 如果所有目标值相等则退出
    if len(set(dataSet[:,-1].tolist())) == 1:
        return None, leaf_type(dataSet)
    m, n = dataSet.shape
    S = err_type(dataSet)
    min_S = np.inf
    best_index = 0; best_value = 0
    for feature_index in range(n-1):
        for feature_value in set(dataSet[:,feature_index]):
            sub_dataSet0, sub_dataSet1 = split_dataSet(dataSet, feature_index, feature_value)
            # 如果剩余样本小于最小样本数，则尝试用下一中方法切分
            if (sub_dataSet0.shape[0] < tolN) or (sub_dataSet1.shape[0] < tolN):
                continue
            new_S = err_type(sub_dataSet0) + err_type(sub_dataSet1)
            if new_S < min_S:
                best_index = feature_index
                best_value = feature_value
                min_S = new_S
    # 如果减少的误差不满足要求
    if (S - min_S) < tolS:
        return None, leaf_type(dataSet)
    sub_dataSet0, sub_dataSet1 = split_dataSet(dataSet, best_index, best_value)
    if (sub_dataSet0.shape[0] < tolN) or (sub_dataSet1.shape[0] < tolN):
        return None, leaf_type(dataSet)
    return best_index, best_value

def create_tree(dataSet, leaf_type=modelLeaf, err_type=modelErr, ops=(1,4)):
    """
    递归创建树
    :param dataSet:
    :param leaf_type:
    :param err_type:
    :param ops:
    :return:
    """
    feature_index,feature_value = choose_best_split(dataSet, leaf_type, err_type, ops)
    # 如果节点不能再分，将该节点保存为叶节点
    if feature_index == None:
        return feature_value
    tree ={}
    tree['feature_index'] = feature_index
    tree['feature_value'] = feature_value
    #分割数据集
    left_dataSet, right_dataSet = split_dataSet(dataSet, feature_index, feature_value)
    # 递归
    tree['left'] = create_tree(left_dataSet, leaf_type, err_type, ops)
    tree['right'] = create_tree(right_dataSet, leaf_type, err_type, ops)
    return tree

def is_tree(obj):
    return type(obj).__name__ == 'dict'

# 预测过程---------------------------------------
def calc_y(W, X_in):
    n = X_in.shape[1]
    X = np.ones((1,n+1))
    X[:,1:n+1] = X_in
    return float(X*W)

def predict_one(tree, X):
    if not is_tree(tree):
        return calc_y(tree, X)
    if X[0,tree['feature_index']] > tree['feature_value']:
        return predict_one(tree['left'], X)
    else:
        return predict_one(tree['right'], X)


def predict(tree, test_data):
    X = np.mat(test_data)
    m = X.shape[0]
    y_hat = np.mat(np.zeros((m,1)))
    for i in range(m):
        y_hat[i,0] = predict_one(tree, X[i])
    return y_hat



if __name__ == '__main__':
    ops = (0.02,5)
    dataSet = load_data()
    tree = create_tree(dataSet, ops=ops)
    print(tree)
    test_data = load_data()
    y_hat = predict(tree, test_data[:,0:-1])
    print(mean_squared_error(y_hat, test_data[:,-1]))