import matplotlib.pyplot as plt
import numpy as np
import models.AdaBoost as adaboost

def plot_ROC(pred_strengths, class_labels):
    # 光标的位置
    cur = (1.0, 1.0)
    y_sum = 0.0
    num_pos_class = sum(np.array(class_labels) == 1)
    y_step = 1 / float(num_pos_class)
    x_step = 1 / float(len(class_labels) - num_pos_class)
    sorted_indicies = pred_strengths.argsort()
    print(sorted_indicies)
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1:
            delX = 0; delY = y_step;
        else:
            delX = x_step; delY = 0;
            y_sum += cur[1]
        print(cur)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    # 绘制对角线
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('FP')
    plt.xlabel('TP')
    ax.axis([0, 1.1, 0, 1.1])
    plt.show()
    # 曲线下的面积，相当于细分成许多小矩形来计算面积
    print('auc:', y_sum*x_step)

if __name__ == '__main__':
    X, y = adaboost.load_data()
    model,agg_calssest = adaboost.adaboost_train_ds(X, y)
    plot_ROC(agg_calssest.T,y)
