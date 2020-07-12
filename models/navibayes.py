from collections import Counter
import numpy as np
import re

def load_data():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1] # 1代表侮辱性言论， 0代表正常
    return posting_list, class_vec

def create_vocab_list(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = set(doc) | vocabSet # 取并集
    return vocabSet

def set_of_word2vec(vocabList, inputSet):
    """
    词集模型 只计算每个词是否出现
    @:param vocabList模型中所有单词的列表
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in vocablist" % word)
    return returnVec

def bag_of_word2vec(vocabList, inputSet):
    """
    词袋模型 计算每个词出现的次数
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word %s is not in vocablist" % word)
    return returnVec

def trainNB0(X, y):
    doc_num = len(X)
    # 每个类别的概率
    labels_count = Counter(y)
    p0 = labels_count.get(0) / doc_num
    p1 = labels_count.get(1) / doc_num
    # 每个类别的单词总数
    p0_denom, p1_denom = 2.0, 2.0   # 防止没出现过的单词概率为0
    # 每个类别每个单词出现的次数
    p0_word_num, p1_word_num = np.ones_like(X[0]),np.ones_like(X[0]) # 防止没出现过的单词概率为0

    for i in range(doc_num):
        if y[i] == 0:
            p0_denom += sum(X[i])
            p0_word_num += X[i]
        else:
            p1_denom += sum(X[i])
            p1_word_num += X[i]
    # 每个类别中每个单词出现的概率
    p0_vect = p0_word_num / p0_denom
    p1_vect = p1_word_num / p1_denom
    # 防止数值下溢
    p0_vect = np.log(p0_vect)
    p1_vect = np.log(p1_vect)
    return p0, p1, p0_vect, p1_vect

def classify(X_onehot, p0_vect, p1_vect, p0, p1):
    p0_predict = sum(X_onehot * p0_vect) + np.log(p0)
    p1_predict = sum(X_onehot * p1_vect) + np.log(p1)
    if p0_predict > p1_predict:
        return 0
    else:
        return 1

def text_parse(text):
    list_of_token = re.split('\\W*', text)
    return [x.lower() for x in list_of_token if len(X)>2]


def spanTest():
    """垃圾邮件分类"""

if __name__ == '__main__':
    X, y= load_data()
    # 得到所有单词
    words = create_vocab_list(X)
    print(X)
    # 得到输入向量的one-hot
    dataSet = set_of_word2vec(list(words), X[0])
    print(dataSet)
    X_onehot = []
    for x in X:
        X_onehot.append(set_of_word2vec(list(words), x))
    p0, p1, p0_vect, p1_vect = trainNB0(X_onehot, y)
    print(p0, p1)
    print(words)
    print(p0_vect)
    print(p1_vect)

    X_test = ['love', 'my', 'garbage']
    X_test = set_of_word2vec(list(words), X_test)
    predict = classify(X_test, p0_vect, p1_vect, p0, p1)
    print(predict)

    # 词袋模型
    mySent = 'this book is the best book on'
    regEx = re.compile('\\W*') # 除单词 数字以外的任意字符
    X = regEx.split(mySent)
