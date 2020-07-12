import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet # 沿Y轴复制dataSetSize个，沿X轴复制1倍（不复制）
    sqDIffMat = diffMat ** 2
    sqDistance = sqDIffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    # 距离从小到大排序
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label,0) + 1
    # 返回前k个点中出现频率最高的label
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, labels = createDataSet()
    predict = knn([0, 0], group, labels, 3)
    print(predict)


