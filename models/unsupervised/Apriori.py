"""
关联分析
"""

import numpy as np

def load_data():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def create_C1(dataSet):
    """创建初始每个元素只有一项集合"""
    C1 = []
    for transcation in dataSet:
        for item in transcation:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frazenset 即不可变集合
    # 对C1的每一项构建一个不变集合, 因为之后需要用这些集合作为字典的键
    return set(map(frozenset, C1))

def scanD(dataSet, Ck, min_support):
    """
    计算每个元素的支持度，并返回符合条件的
    :param D:
    :param Ck: 候选项集列表
    :param min_support:最小支持度
    :return:
    """
    ssCnt = {}
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                if can in ssCnt:
                    ssCnt[can] += 1
                else:
                    ssCnt[can] = 1
    item_nums = float(len(dataSet))
    ret_list = []
    supports = {}
    for key in ssCnt:
        support = ssCnt[key] / item_nums
        if support >= min_support:
            ret_list.insert(0, key)
        supports[key] = support
    return ret_list, supports

def apriori_gen(LK, k):
    retList = []
    LK_len = len(LK)
    for i in range(LK_len):
        for j in range(i+1, LK_len):
            # 减少运算的技巧，将0~K-2部分相同的合并
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j])[:k-2]

            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList

def apriori(dataSet, min_support):
    C1 = create_C1(dataSet)
    L1, supports = scanD(dataSet, C1, min_support)
    K = 2
    L = [L1]
    while len(L[K-2]) > 0:
        CK = apriori_gen(L[K-2], K)
        LK, supports_K = scanD(dataSet, CK, min_support)
        L.append(LK)
        supports.update(supports_K)
        K+=1
    return L, supports



# 关联规则
def calc_conf(freq_set, H, supports, rule_list, min_conf=0.7):
    """
    计算可信度
    公式：support(P -> H) = support(P|H) / support(P)
    :param freq_set:
    :param H:
    :param supports:
    :param br1:
    :param min_conf:
    :return:
    """
    prunde_H = []
    for conSeq in H:
        conf = supports[freq_set] / supports[freq_set - conSeq]
        if conf > min_conf:
            print(freq_set - conSeq, '-->', conSeq, ' conf:', conf)
            # 将符合条件规则放入主函数的规则列表中
            rule_list.append((freq_set - conSeq, conSeq, conf))
            prunde_H.append(conSeq)
    return prunde_H

def rules_from_conseq(freqSet, H, supports, rule_list, min_conf=0.7):
    """
    生成候选规则集合
    :param freqSet:单个频繁项 例：【1，2】
    :param H:
    :param supports:所有频繁项的支持度
    :param rule_list:主函数的规则列表
    :param min_conf:
    :return:
    """
    m = len(H[0])
    # 箭头右侧的H长度最多为频繁项的长度-1
    if (m+1) < len(freqSet):
        # 创建Hm+1条新候选规则
        Hm_p1 = apriori_gen(H, m+1)
        Hm_p1 = calc_conf(freqSet, Hm_p1, supports, rule_list, min_conf)
        # 能够找到的最长的候选规则只会有一个，即和频繁项freqSet相同
        if len(Hm_p1) > 1:
            rules_from_conseq(freqSet, Hm_p1, supports, rule_list, min_conf)

def generate_rules(L, supports, min_conf=0.7):
    """
    关联规则生成的主函数
    :param L:
    :param supports:
    :param min_conf: 最小信任度
    :return:
    """
    rule_list = []
    for i in range(1, len(L)): # 跳过只有一项的平凡项集，因为不能生成关联规则
        for freq_item in L[i]: # 遍历每个频繁项集
            H1 = [frozenset([item]) for item in freq_item]
            if i > 1:
                rules_from_conseq(freq_item, H1, supports, rule_list, min_conf)
            else:
                calc_conf(freq_item, H1, supports, rule_list, min_conf)
    return rule_list

if __name__ == '__main__':
    dataSet = load_data()
    L, supports = apriori(dataSet, 0.5)
    print(L)
    print(supports)
    print('关联规则')
    rules = generate_rules(L, supports, min_conf=0.7)
    print(rules)









