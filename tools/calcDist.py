import numpy as np


def ecludSim(A, B):
    """欧式距离"""
    return 1.0 / (1.0 + np.linalg.norm(A - B))

def pearson_correlation(A, B):
    """皮尔逊相关度"""
    if len(A) < 3:
        return 1.0
    corr = np.corrcoef(A, B)
    # 将相似度归一化到 0-1
    return 0.5 + 0.5 * corr[0][1]

def cosine_similiarty(A, B):
    """计算余弦相似度"""
    sim = A*B.T / (np.linalg.norm(A) * np.linalg.norm(B))
    return 0.5 + 0.5 * sim

if __name__ == '__main__':
    a = np.mat([1, 2, 3])
    b = np.mat([4, 5, 6])
    print(cosine_similiarty(a, b))
    print(a.T*b)