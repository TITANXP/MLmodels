from math import log
from collections import Counter
import matplotlib.pyplot as plt

def create_dataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    tags = ['no surfacing', 'flippers'] # 两个特征的标签
    return dataSet, tags

def cal_shannon_ent(data_labels):
    """计算香农熵"""
    data_len = len(data_labels)
    # 创建字典记录每个label的个数
    labelCounts = {}
    for i in range(data_len):
        label = data_labels[i]
        if label not in labelCounts:
            labelCounts[label] = 0
        labelCounts[label] += 1
    shannonEnt = 0.0
    for label in labelCounts:
        prob = float(labelCounts[label] / data_len)
        shannonEnt += - prob * log(prob, 2)
    return shannonEnt

def split_dataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param data: 数据集
    :param axis: 划分数据集的特征
    :param value: 返回划分后特征为该值的数据
    """
    new_dataSet = []
    for x in dataSet:
        if x[axis] == value:
            temp = x[:axis]
            temp.extend(x[axis+1:])
            new_dataSet.append(temp)
    return new_dataSet

def choose_best_split_feature(dataSet):
    """选择信息增益最大的feature"""
    num_features = len(dataSet[0]) - 1
    base_shannon_entropy = cal_shannon_ent([x[-1] for x in dataSet])
    best_info_gain = 0
    best_split_feature = -1
    for i in range(num_features):
        # 得到划分数据集的feature的唯一值
        feature_values = [example[i] for example in dataSet]
        feature_values = set(feature_values)
        new_shonnon_entropy = 0.0
        for feature_value in feature_values:
            # 计算每种划分方式的信息熵
            sub_dataSet = split_dataSet(dataSet, i, feature_value)
            prob = len(sub_dataSet) / float(len(dataSet))
            new_shonnon_entropy += prob * cal_shannon_ent([x[-1] for x in sub_dataSet])
        # 计算信息增益
        info_gain = base_shannon_entropy - new_shonnon_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_split_feature = i
    return best_split_feature

def ID3(dataSet, tags):
    print(dataSet)
    # label完全相同则停止划分
    labels = [x[-1] for x in dataSet]
    if len(set(labels)) == 1:
        return labels[0]
    # 没有特征可以分时停止
    if len(dataSet[0]) == 1:
        return Counter(labels)[0][0] # 返回数量最多的label
    best_feature = choose_best_split_feature(dataSet)
    best_feature_tag = tags[best_feature]
    tree = {best_feature_tag : {}}
    del(tags[best_feature])
    feature_values = [x[best_feature] for x in dataSet]
    feature_values = set(feature_values)
    for value in feature_values:
        sub_tags = tags[:]
        tree[best_feature_tag][value] = ID3(split_dataSet(dataSet, best_feature, value), sub_tags)
    return tree

def plot_tree(tree):
    # 定义样式
    decision_node = dict(boxstyle='sawtooth', fc='0.8')
    leaf_node = dict(boxstyle='round4', fc='0.8')
    arrow_args = dict(arrowstyle='<-')

    def plot_node(node_txt, center_pt, parent_pt, node_type):
        """
        :param node_txt:
        :param center_pt:节点坐标
        :param parent_pt: 父节点坐标
        :param node_type:
        :return:
        """
        create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va='center', ha='center', bbox=node_type, arrowprops=arrow_args)

    def plot_mid_text(center_pt, parent_pt, text):
        x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
        y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]

    def plot_tree(tree, parent_pt, node_txt):
        leaf_num = get_leaf_num(tree)
        max_depth = get_max_depth(tree)
        first_str = list(tree.keys())[0]
        center_pt = (plot_tree.xOff +(1.0 + float(leaf_num))/2.0/plot_tree.totalW, plot_tree.yOff)

        plot_mid_text(center_pt, parent_pt, node_txt)
        plot_node(first_str, center_pt, parent_pt, decision_node)
        second_dict = tree[first_str]
        plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                plot_tree(second_dict[key], center_pt, str(key))
            else:
                plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
                plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), center_pt, leaf_node)
                plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_pt, str(key))
        plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD

    def create_plot():
        fig = plt.figure(1,facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
        plot_tree.totalW = float(get_leaf_num(tree))
        plot_tree.totalD = float(get_max_depth(tree))
        plot_tree.xOff = -0.5 / plot_tree.totalW
        plot_tree.yOff = 1.0
        # plot_node('决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
        # plot_node('叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
        plot_tree(tree, (0.5, 1.0),'')
        plt.show()
    create_plot()

def get_leaf_num(tree):
    leaf_num = 0
    first_str = list(tree.keys())[0]
    second_dict= tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            leaf_num += get_leaf_num(second_dict[key])
        else:
            leaf_num+=1
    return leaf_num

def get_max_depth(tree):
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            new_depth = 1 + get_max_depth(second_dict[key])
        else:
            new_depth = 1
        if new_depth > max_depth:
            max_depth = new_depth
    return max_depth

def classify(tree, tags, X):
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    feature_index = tags.index(first_str)
    for key in second_dict.keys():
        if X[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                label = classify(second_dict[key], tags,X)
            else:
                label = second_dict[key]
    return label

def save_tree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()

def load_tree(filename):
    import pickle
    fr = open(filename, 'r')
    return pickle.load(fr)

if __name__ == '__main__':
    dataSet, tags = create_dataSet()
    shannonEnt = cal_shannon_ent([x[-1] for x in dataSet])
    # print(shannonEnt)
    # best_feature = choose_best_split_feature(dataSet)
    # print(best_feature)
    tree = ID3(dataSet, tags.copy())
    print(tree)
    plot_tree(tree)
    print(get_leaf_num(tree))
    print(get_max_depth(tree))
    print(classify(tree, tags, [1,1]))

