import numpy as np

def load_data():
    return [
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1]
    ]

if __name__ == '__main__':
    data = load_data()
    U, sigma, VT = np.linalg.svd(data)
    print('sigma最后两个值太小，所以可以去掉')
    print(sigma)
    # 重构原始矩阵，先构建一个3x3的矩阵
    sig3 = [
        [sigma[0], 0, 0],
        [0, sigma[1], 0],
        [0, 0, sigma[2]]
    ]
    # 因为只是用了seima的前三个奇异值， 所以只需使用U的前三列，和VT的前三行
    data_s = np.mat(U[:,:3]) * np.mat(sig3) * np.mat(VT[:3])
    print(data_s)
    # 如何确定要保留几个奇异值？
    # （1）保留矩阵中90%的信息：为了计算总信息，需要求所有奇异值的平方和，再将奇异值的平方和累加到90%以上
    # （2）当矩阵中有上万的奇异值时，保留前面的2000或3000个

