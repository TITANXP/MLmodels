import numpy as np
import matplotlib.pyplot as plt

def load_data():
    x = np.random.rand(100, 1)
    noise = (np.random.rand(100,1) - 0.5)/ 5
    y = 1.5 * x + noise
    # data = [
    #     [1, 10],
    #     [2, 22],
    #     [5, 49],
    #     [6, 62],
    #     [6, 60],
    #     [10, 99]
    # ]
    data = np.hstack((x, y))
    return np.mat(data)

def pca(dataSet, topN=999):
    # 1.减去平均值
    mean_values = np.mean(dataSet, axis=0)
    dataSet = dataSet - mean_values
    # 2.找到是的数据方差最大的坐标轴
    #   根据推导，方差等于数据集协方差的特征值，所以找到最大的特征值λ，对应的特征向量即为要找的坐标轴
    # （1）求协方差矩阵
    #       [[10.4, 102.2],                   [cov(x1,x1), cov(x1,x2)]
    #       [102.2, 1005.86666667]])          [cov(x2,x1), cov(x2, x2)]
    covMat = np.cov(dataSet, rowvar=0)
    # （2）找到所有特征值和特征向量
    # α^t Σ α = λ
    eig_vals, eig_vects = np.linalg.eig(covMat)
    # (3)排序
    eig_vals_sort_index = np.argsort(eig_vals)
    #    取topN个
    eig_vals_sort_index = eig_vals_sort_index[:-(topN+1):-1] # 反向打印,取index=-（topN+1) (不包含） 到
    red_eig_vects = eig_vects[:,eig_vals_sort_index]
    print(eig_vals_sort_index)
    print(red_eig_vects)
    # 3.将数据转换到新空间 nx2 2x1 = nx1
    lowerData = dataSet * red_eig_vects
    # 4.转换到原空间 nx1 1x2 = nx2
    reconData = (lowerData * red_eig_vects.T) + mean_values
    return lowerData, reconData

def plot_data(lowerData, data):
    fig = plt.figure()
    print(data[:,0].flatten().A[0])
    plt.scatter(data[:,0].flatten().A[0], data[:,1].flatten().A[0], c='red')
    plt.scatter(lowerData[:,0].flatten().A[0], lowerData[:,1].flatten().A[0], c='blue')
    plt.show()


if __name__ == '__main__':
    data = load_data()
    lowerData, reconData = pca(data, topN=1)
    plot_data(reconData, data)