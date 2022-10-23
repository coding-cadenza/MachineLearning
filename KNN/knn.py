from statistics import mode

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score

from kd_tree import KDTree

plt.rcParams['font.sans-serif'] = ['SimHei']


class KNN:
    def __init__(self, values, labels):
        self.values = np.array(values)
        self.labels = np.array(labels)
        self.k = 1  # 初始化k值
        self.min_correct_rate = 0.83  # 能接受的最小准确率,自行调整
        self.train()  # 训练出k值
        self.kd_tree = KDTree(self.values, self.labels)  # 创建全数据kd树

    '''训练主要目的是为了选取k值'''

    def train(self):
        train_len = int(len(self.values) * 0.7)  # 70%做训练集
        # 训练集
        train_values = self.values[:train_len]
        train_labels = self.labels[:train_len]

        # 验证集
        verify_values = self.values[:train_len + 1]
        verify_labels = self.labels[:train_len + 1]

        # 创建kd树
        self.kd_tree = KDTree(train_values, train_labels)

        # 调参
        correct_rate = 0
        while correct_rate < self.min_correct_rate:
            res_labels = self.predict(verify_values)
            # 计算准确率
            correct_rate = self.cal_correct_rate(verify_labels, res_labels)
            self.k += 1

        print("训练完成，验证集准确率为:{0},k为:{1}".format(correct_rate, self.k))

    '''
    knn入口
    :param target: 目标点，一个或者多个
    '''

    def predict(self, target):
        if target is None:
            return

        target = np.array(target)
        shape = target.shape
        if len(shape) == 1:  # 只有一个实例
            return self.predict_core(target)
        else:
            res = []
            for i in range(shape[0]):
                res.append(self.predict_core(target[i]))
            res = np.array(res)
            return res

    '''
    knn的核心方法
    :param target: 这里target只能是一个实例
    '''

    def predict_core(self, target):
        # 获取k个最邻近点对应的标签
        knn_labels = self.kd_tree.search_KNN(target, self.k)[1]
        # 取出k个点中最多的类别最为答案
        return mode(knn_labels)

    '''
    precision_score老报错,自己写一个计算准确率
    origin可以互换predict
    '''

    def cal_correct_rate(self, origin, predict):
        Len = len(origin)
        count = 0
        for i in range(Len):
            if origin[i] == predict[i]:
                count += 1
        return count / Len


if __name__ == '__main__':
    # 读取数据
    df = pd.DataFrame(pd.read_csv('./data/knn_data_2')) # 二维鸢尾花数据
    # df = pd.DataFrame(pd.read_csv('./data/knn_data_3'))  # 三维鸢尾花

    values = df.values[:, :3]
    labels = df.values[:, -1]
    train_len = int(len(values) * 0.9)

    # 测试集
    test_values = values[:train_len]
    test_labels = labels[:train_len]
    # 训练集
    train_values = values[:train_len + 1]
    train_labels = labels[:train_len + 1]

    # 二维数据下的散点图
    # x1_min, x1_max = values[:, 0].min() - 0.5, values[:, 0].max() + 0.5  # 第一维坐标最大最小值
    # x2_min, x2_max = values[:, 1].min() - 0.5, values[:, 1].max() + 0.5  # 第二维坐标最大最小值
    # plt.scatter(values[:, 0], values[:, 1], c=labels, edgecolor="k")
    # plt.xlim(x1_min, x1_max)
    # plt.ylim(x2_min, x2_max)
    # plt.xlabel("特征1")
    # plt.ylabel("特征2")
    # plt.show()

    knn = KNN(train_values, train_labels)
    res = knn.predict(test_values)
    correct_rate = knn.cal_correct_rate(test_labels, res)
    print('测试集预测准确率为:{}'.format(correct_rate))
