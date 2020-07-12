"""
二分均值聚类
    为了解决kmeans收敛于局部极小值的问题
        首先将所有点看成一个簇，然后将该簇一分为二
        然后选择一个划分后可以最大程度降低误差的簇继续进行划分
        直到达到k
"""
import numpy as np
import matplotlib.pyplot as plt
from models.unsupervised import kmeans

def load_data():
    centroid = [[1, 1], [5, 16], [14, 1.5]]
    noise = np.random.randn(50, 2)
    X = []
    for c in centroid:
        x = noise + c
        X.extend(x)
    X = np.array(X)
    return X

def calc_dist(X1, X2):
    """计算欧氏距离"""
    return np.sqrt(sum(np.power((X1 - X2), 2)))

def plot_cluster(X, centroids, point_cluster):
    centroids = np.array(centroids)
    # color_list = ['lightpink','azure', 'turquoise', 'cyan', 'skyblue', 'violet']
    for i in range(len(centroids)):
        # color = color_list[i % len(color_list)]
        plt.scatter(X[point_cluster[:,0] == i][:,0], X[point_cluster[:,0] == i][:,1])
        plt.scatter(centroids[i,0], centroids[i,1], marker='D', s=100, c='red')
    plt.show()

def bin_kmeans(X, k):
    m, n = X.shape
    point_cluster = np.zeros((m, 2))
    # 创建一个初始簇
    centroid_0 = np.mean(X, axis=0).tolist()[0]
    centroids = [centroid_0]
    for i in range(m):
        print(np.power(calc_dist(X[i], centroid_0), 2))
        point_cluster[i, 1] = np.power(calc_dist(X[i], centroid_0), 2)
    # 当簇数小于k
    while len(centroids) < k:
        min_error = np.inf
        # 尝试划分每一个簇
        for c in range(len(centroids)):
            print(point_cluster)
            new_centroids, new_point_cluster = kmeans.kmeans(X[point_cluster[:,0] == c], 2)
            new_error = sum(new_point_cluster[:, 1])
            # 没有进行划分的簇的总误差
            error_not_split = sum(point_cluster[point_cluster[:,0] != c][:,1])
            if (new_error + error_not_split) < min_error:
                best_centroid_to_split = c
                best_centroids = new_centroids
                best_point_cluster = new_point_cluster.copy()
                min_error = new_error + error_not_split
        # 根据结果重新分配簇
        print('the best centroid to split is:', c)
        del(centroids[best_centroid_to_split])
        centroids.extend(best_centroids.tolist())
        point_cluster[point_cluster[:,0] == best_centroid_to_split] = best_point_cluster
    return centroids, point_cluster


if __name__ == '__main__':
    X= load_data()
    centroids, point_cluster = bin_kmeans(X, 3)
    plot_cluster(X, centroids, point_cluster)



