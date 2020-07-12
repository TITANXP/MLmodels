import numpy as np
import matplotlib.pyplot as plt

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

def rand_centroid(X, k):
    """随机初始化质心"""
    n = X.shape[1]
    centroids = np.mat(np.zeros((k,n)))
    for i in range(n):
        min_v = min(X[:,i])
        max_v = max(X[:,i])
        centroids[:,i] = min_v + (max_v - min_v) * np.random.rand(k, 1)
    return centroids

def plot_cluster(X, centroids, point_cluster):
    # color_list = ['lightpink','azure', 'turquoise', 'cyan', 'skyblue', 'violet']
    for i in range(len(centroids)):
        # color = color_list[i % len(color_list)]
        plt.scatter(X[point_cluster[:,0] == i][:,0], X[point_cluster[:,0] == i][:,1])
        plt.scatter(centroids[i,0], centroids[i,1], marker='D', s=100, c='red')
    plt.show()


def kmeans(X, k, dist_func=calc_dist, init_centriod_func=rand_centroid):
    m, n = X.shape
    cluster_changed = True # 质心是否改变
    centroids = init_centriod_func(X, k)
    # centroids = np.mat( [[1, 1], [5, 16], [14, 1.5]])
    point_cluster = np.zeros((m,2)) # 距离最近的质心index，以及最短的距离平方
    while cluster_changed:
        cluster_changed = False
        # 为每个点找到距离最近的重心
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for c in range(k):
                new_dist = calc_dist(centroids.tolist()[c], X[i])
                if new_dist < min_dist:
                    min_dist = new_dist
                    min_index = c
            if point_cluster[i,0] != min_index:
                point_cluster[i] = min_index, min_index**2
                print('------------')
                cluster_changed = True
        # 重新计算质心
        print(centroids)
        for c in range(k):
            centroids[c] = X[point_cluster[:,0] == c].mean(axis=0)
    return centroids, point_cluster

if __name__ == '__main__':
    X = load_data()
    centroids, point_cluster = kmeans(X, 3)
    plot_cluster(X, centroids, point_cluster)



