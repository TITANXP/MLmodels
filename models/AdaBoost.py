import numpy as np

def load_data():
    X = np.mat(
        [
            [1., 2.1],
            [2, 1.1],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]
        ]
    )
    y = [1, 1, -1, -1, 1]
    return X, y

def stump_classify(X, dimen, thresh_val, thresh_ineq):
    """
    通过比较阈值对数据进行分类
    :param X:
    :param dimen:
    :param thresh_val:
    :param thresh_ineq:
    :return:
    """
    ret_array = np.ones((np.shape(X)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[X[:,dimen] <= thresh_val] = -1.0
    else:
        ret_array[X[:,dimen] > thresh_val] = -1.0
    return ret_array




def build_stump(X, y, D):
    """
    根据当前权重找出一个error最小的决策树桩
    :param X:
    :param y:
    :param D: 每个样本的权重
    :return:
    """
    X = np.mat(X)
    y = np.mat(y).T
    m,n = np.shape(X)
    num_steps = 10
    best_classest = np.mat(np.zeros((m,1)))
    best_stump = {}
    min_error = np.inf
    for i in range(n): # 遍历每个特征
        range_min = X[:,i].min()
        range_max = X[:,i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(num_steps):
            for inequal in ['lt', 'gt']:
                thresh_val = range_min + j * step_size
                predict_vals = stump_classify(X, i, thresh_val, inequal)
                err_arr = np.ones((m, 1))
                err_arr[predict_vals == y] = 0
                weighted_err = D.T * err_arr
                if weighted_err < min_error:
                    min_error = weighted_err
                    best_classest = predict_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_classest

def adaboost_train_ds(X, y, num_iter = 90):
    """
    adaboost的主要流程
    :param X:
    :param y:
    :param num_iter:
    :return:
    """
    weak_class = []
    m = np.shape(X)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_classEst = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        best_stump, error, classest = build_stump(X, y, D)
        print('D:', D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class.append(best_stump)
        print('classEst:', classest)
        expon = np.multiply((-1 * alpha * np.mat(y).T), classest)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 错误率累加
        agg_classEst += alpha * classest
        agg_errors = np.multiply(np.sign(agg_classEst) != np.mat(y).T, np.ones((m,1)))
        error_rate = agg_errors.sum() / m
        print('total error:', error_rate)
        if error_rate == 0:
            break
    return weak_class, agg_classEst

def classify(X, stumps):
    """
    进行预测
    :param X:
    :param stumps: 学习得到的所有决策树桩
    """
    X = np.mat(X)
    m = np.shape(X)[0]
    predicts = np.zeros((m, 1))
    for i in range(len(stumps)):
        predict = stump_classify(X, stumps[i]['dim'], stumps[i]['thresh'], stumps[i]['ineq'])
        predicts += stumps[i]['alpha'] * predict
    return np.sign(predicts)




if __name__ == '__main__':
    X,y = load_data()
    model, _ = adaboost_train_ds(X, y)
    print(model)
    predict = classify(X, model)
    print("predict:\n", predict)