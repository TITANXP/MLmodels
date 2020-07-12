import numpy as np
from tools.calcDist import pearson_correlation

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


def standEst(dataSet, user, dist_func, item):
    """
    基于物品相似度的推荐引擎
    :param dataSet:
    :param user:
    :param dist_func:
    :param item:
    :return:
    """
    user_num, item_num = dataSet.shape
    similar_sum = 0 # 要评分的物品和所有符合条件的物品的相似度和
    rate_similat_sum = 0 # 估计的评分和
    for i in range(item_num):
        userRating = dataSet[user,i]
        # 由于基于物品，所以此用户没有评分的物品不能用
        if userRating == 0:
            continue
        # 过滤掉没有对这两个物品评分的用户，用对两个物品评分过的用户的评分，作为这两个物品的特征
        overLap = np.nonzero(np.logical_and(dataSet[:,item].A>0, dataSet[:,i].A>0))
        # 计算两个物品的相似度
        #   如果没有用户对这两个物品都进行了评分，则两个物品相似度记为0
        if len(overLap) == 0:
            similar = 0
        else:
            similar = dist_func(dataSet[overLap,item], dataSet[overLap, i])
            print(similar)

        # 将相似度作为权重
        similar_sum += similar
        rate_similat_sum += similar * userRating
    if similar_sum == 0:
        return 0
    else:
        # 加权平均
        return rate_similat_sum / similar_sum

def svdEst(dataSet, user, dist_func, item):
    """
    基于svd的评分估计
    :param dataSet:
    :param user:
    :param dist_func:
    :param item:
    :return:
    """
    dataSet = np.mat(dataSet)
    user_num, item_num = dataSet.shape
    similar_sum = 0
    rate_similar_sum = 0
    U, sigma, VT = np.linalg.svd(dataSet)
    sigma4 = np.mat(np.eye(4) * sigma[:4])
    # 降维 构建转换后的物品 nxm   mx4 4x4
    xformedItems = dataSet.T * U[:,:4] * sigma4.I
    print('shape',xformedItems.shape)
    for i in range(item_num):
        userRating = dataSet[user,i]
        if userRating == 0 or i == item:
            continue
        similar = dist_func(xformedItems[item], xformedItems[i])
        similar_sum += similar
        rate_similar_sum += similar * userRating
    if similar_sum == 0:
        return 0
    else:
        return rate_similar_sum / similar_sum



def recommend(dataSet, user, N=3, dist_func=pearson_correlation, est_func=standEst):
    """

    :param dataSet:
    :param user:
    :param N:
    :param dist_func: 计算距离的函数
    :param est_func: 估计函数
    :return:
    """
    # 寻找未评级的物品
    dataSet = np.mat(dataSet)
    un_rated_items = np.nonzero(dataSet[user].A==0)[1] # nonezero返回的非零元素的列号
    if len(un_rated_items) == 0:
        return '所以物品都已评分'
    item_scores = []
    for item in un_rated_items:
        estimated_score = est_func(dataSet, user, dist_func, item)
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]

if __name__ == '__main__':
    data = load_data()
    user =2
    rating = recommend(data, user)
    for i in rating:
        print('用户', user, ' 物品', i[0], ' 评分：',i[1])

    # 基于svd的评分估计
    rating = recommend(data, user, est_func=svdEst)
    print(rating)


