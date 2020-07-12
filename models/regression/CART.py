"""
---- 回归树： 每个叶子节点是单个值
ID3的问题
    1.切分过于迅速
        ID3每次选取当前最佳的特征来分割数据，并按照该特征的所有可能取值来切分
        一旦按照某个特征切分后，该特征在之后的算法执行过程中将不会再起作用，这种切分方法过于迅速
    2.不能处理连续值
        只有事先将连续型特征转换成离散值，才能在ID3中使用，但这回破环连续型变量的内在性质
CART使用二元切分法来处理连续型特征
    每次把数据切分成两份，如果特征值等于切分所要求的值，则这些数据进入左子树，反之进入右子树，二元切分法节省了树的构建时间。
对CART稍作修改就可以处理回归问题
    ID3用香农熵来度量结合的无组织程度， 如果选用其它方法来代替香农熵，就可以使用树来处理回归问题。
        首先计算所有数据的均值，然后计算每个数据和均值的差，为了正负差值同等看待，一般使用绝对值或平方来代替上述差值
            与方差的区别：方差是平方误差的均值，而这里的是平方误差的总值

剪枝
    1.预剪枝： 在choose_best_split()中的提前终止条件
            缺点：对输入参数tolS和tolN过于敏感（如果把数据集放大100倍会得到不同的树）
    2. 后剪枝： 将数据集分为测试集和训练集
        步骤：
            （1）构建树
            （2）从上而下找到叶节点，用测试集判断将这些叶节点合并是否会降低测试误差，如果是就合并
"""

import numpy as np

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

def regLeaf(dataSet):
    """创建叶节点"""
    return dataSet[:,-1].mean()

def regErr(dataSet):
    """计算label的总方差"""
    return dataSet[:,-1].var() * dataSet.shape[0]

def choose_best_split(dataSet, leaf_type=regLeaf, err_type=regErr, ops=[1,4]):
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
            new_S = err_type(sub_dataSet0) + err_type(sub_dataSet1)
            # 如果剩余样本小于最小样本数，则尝试用下一中方法切分
            if sub_dataSet0.shape[0] < tolN or sub_dataSet1.shape[0] < tolN:
                continue
            if new_S < min_S:
                best_index = feature_index
                best_value = feature_value
                min_S = new_S
    # 如果减少的误差不满足要求
    if (S - min_S) < tolS:
        return None, leaf_type(dataSet)
    sub_dataSet0, sub_dataSet1 = split_dataSet(dataSet, best_index, best_value)
    if sub_dataSet0.shape[0] < tolN or sub_dataSet1.shape[0] < tolN:
        return None, leaf_type(dataSet)
    return best_index, best_value

def create_tree(dataSet, leaf_type=regLeaf, err_type=regErr, ops=(1,4)):
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


# 后剪枝---------------------------------------
def is_tree(obj):
    """判断是树还是叶子节点"""
    return type(obj).__name__ == 'dict'

def get_mean(tree):
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData=None):
    """根据测试集进行后剪枝"""
    # 如果已经没有数据，对树进行塌陷处理
    if testData.shape[0] == 0:
        return get_mean(tree)
    if is_tree(tree['left']) or is_tree(tree['right']):
        left_dataSet, right_dataSet = split_dataSet(testData, tree['feature_index'], tree['feature_value'])
    # 如果是子树，继续剪枝
    if is_tree(tree['left']):
        prune(tree['left'], left_dataSet)
    if is_tree(tree['right']):
        prune(tree['right'], right_dataSet)
    # 如果已经全为叶子节点，则尝试剪枝
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left_dataSet, right_dataSet = split_dataSet(testData, tree['feature_index'], tree['feature_value'])
        error_no_merge = np.sum((left_dataSet[:,-1] - tree['left']) ** 2) + np.sum((right_dataSet[:,-1] - tree['right']) ** 2)
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = np.sum((testData[:,-1] - tree_mean) ** 2)
        if error_merge < error_no_merge:
            print('merge')
            return tree_mean
        else:
            return tree
    else:
        return tree

if __name__ == '__main__':
    dataSet = load_data()
    tree = create_tree(dataSet)
    print(tree)
    test_data = load_data()
    tree = prune(tree, test_data[:10])
    print(tree)
