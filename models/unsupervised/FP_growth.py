

# 创建数据集
def load_simple_data():
    simpData = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simpData

def create_init_dataSet(dataSet):
    retDIct = {}
    for trans in dataSet:
        retDIct[frozenset(trans)] = 1
    return retDIct

# 定义FP树
class treeNode:
    def __init__(self, name, num, parentNode):
        self.name = name
        self.count = num
        # 指向相同元素的节点
        self.nodeLink = None
        self.parent = parentNode
        self.child = {}

    def inc(self, num):
        self.count += num

    def disp(self, ind=1):
        """显示树"""
        print(' '*ind, self.name, ' ', self.count)
        for c in self.child.values():
            c.disp(ind+1)

def create_tree(dataSet, min_sup):
    """

    :param dataSet: {{'h', 'z', 'r', 'p', 'j'}: 1, {'z', 'v', 'u', 's', 'x', 'w', 'y', 't'}: 1}
    :param min_sup:
    :return:
    """
    # 头指针表，每一项出现次数
    header_tabel = {}
    for data in dataSet:
        for item in data:
            header_tabel[item] = header_tabel.get(item, 0) + dataSet[data]
    # 移除不满足最小支持度的项
    htable_keys = set(header_tabel.keys())
    for k in htable_keys:
        if header_tabel[k] < min_sup:
            del header_tabel[k]
    # 构建FP树
    freq_item_set = set(header_tabel.keys())
    #   如果没有元素项满足要求，退出
    if len(freq_item_set) == 0:
        return None, None
    for k in header_tabel:
        header_tabel[k] = [header_tabel[k], None]
    ret_tree = treeNode('NUll set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freq_item_set:
                localD[item] = header_tabel[item][0]
            if len(localD) > 0:
                # 降序
                sorted_items = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
                update_tree(sorted_items, ret_tree, header_tabel, count)
    return ret_tree, header_tabel

def update_tree(items, tree, header_table, count):
    """
    增长树
    :param items:['x', 't']
    :param tree:
    :param header_table:
    :param count:
    :return:
    """
    if items[0] in tree.child:
        tree.child[items[0]].inc(count)
    else:
        tree.child[items[0]] = treeNode(items[0], count, tree)
        if header_table[items[0]][1] == None:
            header_table[items[0]][1] = tree.child[items[0]]
        else:
            update_header(header_table[items[0]][1], tree.child[items[0]])

    if len(items) > 1:
        update_tree(items[1::], tree.child[items[0]], header_table, count)

def update_header(node_to_test, target_node):
    while node_to_test.nodeLink != None:
        node_to_test = node_to_test.nodeLink
    node_to_test.nodeLink = target_node

# ----------------发现前缀路径--------------
def ascend_tree(leafNode, prefixPath):
    """
    从叶子节点向上遍历整棵树，得到路径
    :param leafNode:
    :param prefixPath:
    :return:
    """
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascend_tree(leafNode.parent, prefixPath)

def find_prefix_tree(basePat, treeNode):
    """

    :param basePat: 'x'
    :param treeNode:
    :return:
    """
    condPats = {}
    while treeNode.nodeLink != None:
        prefixPath = []
        ascend_tree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mine_tree(tree, header_table, minSup, preFix, freqItemList):
    """
    递归查找频繁项集
    :param tree:
    :param header_table:
    :param minSup:最小支持度
    :param preFix:set([])
    :param freqItemList: 用来存储频繁项集的空列表
    :return:
    """
    # 1.对头指针表的元素升序排序
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]
    # 2.从条件模式基来构建FP树
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = find_prefix_tree(basePat, header_table[basePat][1])
        myCondTree, myHead = create_tree(condPattBases, minSup)
        if myHead != None:
            mine_tree(myCondTree, myHead, minSup, newFreqSet, freqItemList)




if __name__ == '__main__':
    data = load_simple_data()
    init_data = create_init_dataSet(data)
    print('数据集', init_data)
    FP_tree, header_table = create_tree(init_data, 3)
    print('头指针链表', header_table)
    FP_tree.disp()
    # 抽取条件模式基
    condPats = find_prefix_tree('y', header_table['y'][1])
    # print(condPats)
    freq_items = []
    mine_tree(FP_tree, header_table, 3, set([]), freq_items)
    print('频繁项集', freq_items)
