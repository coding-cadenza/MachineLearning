from collections import Counter

import numpy as np
import pandas as pd


'''
朴素贝叶斯分类器
输出空间由构造函数的labels得出，所以不会预测出训练集不存在的类型
而输入空间由训练的values得出，如果预测的输入的某个特征值在values中不存在，则用拉普拉斯修正
训练过程中也用拉普拉斯训练
'''
class NaiveBayes:
    '''
    构造函数
    :param values: 训练实例
    :param values: 训练标签
    '''

    def __init__(self, values, labels):
        self.values = np.array(values)
        self.labels = np.array(labels)
        self.labels_space = set(self.labels)  # 不重复的标签值
        self.lam = 1  # 拉普拉斯修正
        self.train()  # 开始训练

    def train(self):
        # 变量初始化
        sample_len = self.values.shape[0]  # 样本数
        feature_len = self.values.shape[1]  # 特征数
        pri_p = {}  # 先验概率，key是类，value是值
        con_p = {}  # 条件概率，key是:(类别,特征，特征值)
        label_num_dict = dict(Counter(self.labels))  # 各种标签的数量
        self.label_num_dict = label_num_dict
        # 计算每个维度的取值总数,供拉普拉斯修正使用
        S = {}
        for j in range(feature_len):
            S[j] = len(set(self.values[:, j]))
        self.S = S

        # 经过修正的先验概率
        # 先计算数量再除以总数得到概率，避免精度损失过大
        for key in label_num_dict:
            pri_p[key] = (label_num_dict[key] + self.lam) / (sample_len + len(self.labels_space) * self.lam)

        # 计算条件概率,经过修正
        con_p_mol_dict = {}  # 某种类下某个特征某个特征值的数量
        for i in range(sample_len):
            for j in range(feature_len):
                key = (self.labels[i], j, self.values[i][j])
                if key in con_p_mol_dict:
                    con_p_mol_dict[key] += 1
                else:
                    con_p_mol_dict[key] = 1

        for key in con_p_mol_dict:
            con_p[key] = (con_p_mol_dict[key] + self.lam) / (label_num_dict[key[0]] + self.lam * S[key[1]])

        # 保存到self种
        self.pri_p = pri_p
        self.con_p = con_p

    '''
    朴素贝叶斯预测
    :param value:预测的实例
    '''

    def predict(self, value):
        res = {}  # 每个类计算的比较公式结果
        for label in self.labels_space:
            tmp = self.pri_p[label]  # 初始化为先验概率
            # 条件概率连乘
            for j in range(len(value)):
                # 构造条件概率的key
                key = (label, j, value[j])
                # 对于不存在的条件概率，需要使用拉普拉斯修正
                if key not in self.con_p:
                    tmp *= self.lam / (self.label_num_dict[key[0]] + self.lam * self.S[key[1]])
                else:
                    tmp *= self.con_p[key]

            res[label] = tmp

        # 排序返回概率最大的
        # print('计算结果:', res)
        return sorted(res.items(), key=lambda x: x[1], reverse=True)[0][0]

#
# if __name__ == '__main__':
#     df = pd.DataFrame(pd.read_excel('./data/data_in_book.xlsx'))
#     values = df.values[:, :-1]
#     labels = df.values[:, -1]
#     nb = NaiveBayes(values, labels)
#     print("[2,'S']的预测的类别是:{}".format(nb.predict([2,'S'])))
