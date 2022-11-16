import math
import random

import numpy as np
import pandas as pd
from graphviz import Digraph



'''
结点结构
{
    'split_feat':,# 切分特征 (非叶节点存在)
    'split_value':,# 切分特征值 (非叶节点存在)
    'left':{}, # 左子树 (非叶节点存在)
    'right':{}, # 右子树 (非叶节点存在)
    'label':, # 类别 (所有结点均存在)
    'is_leaf':True, # 叶节点标识，仅叶结点存在
}
'''
class CART:
    def __init__(self, values, labels, features):
        self.values = values
        self.labels = list(labels)
        self.unique_labels = list(set(labels))
        self.features = list(features)
        self.gini_threshold = 0.7 # 基尼指数阈值
        self.root = self.create_tree(self.values, self.labels, self.features.copy())

    '''
    创建决策树
    :param value: 训练实例
    :param labels: 实例对应的类别
    :param features: 实例特征对应名称
    '''

    def create_tree(self, values, labels, features):

        # 数据类别相同，直接返回该类别
        if len(set(labels)) == 1:
            return {'label': labels[0],'is_leaf':True}

        # 特征已用完时，不能再分，返回最多的类别
        if len(features) == 0:
            return {'label': max(labels, key=labels.count),'is_leaf':True}

        # 选择最优特征以及最优特征的最优特征值
        best_feat, best_feat_val = self.choose_best_feat(values, labels)

        # 选不出最优特征时返回(在选最优特征的时候可以剪枝，或者每个特征都只有一种取值了也是选不出来的)
        if best_feat == -1:
            return {'label': max(labels, key=labels.count),'is_leaf':True}
        best_feat_name = features[best_feat]

        # 创建根节点，以最优特征为键名，其孩子为键值(嵌套)
        root = {'split_feat': best_feat_name, 'split_value': best_feat_val,'label': max(labels, key=labels.count)}

        # 二分数据集，递归创建决策树
        del features[best_feat]  # 删去已选特征
        values_equal, labels_equal, values_unequal, labels_unequal = self.split_data(values, labels, best_feat,
                                                                                     best_feat_val)  # 二分数据集

        features_copy = [feature for feature in features]  # 引用冲突解除
        # 创建左右子树
        root['left'] = self.create_tree(values_equal, labels_equal, features)
        root['right'] = self.create_tree(values_unequal, labels_unequal, features_copy)

        return root

    '''
    选出最优特征以及最优切分点的值
    :param values: 实例
    :param labels: 类别 
    '''

    def choose_best_feat(self, values, labels):
        best_feat = -1
        best_feat_val = ''
        exam_num = float(len(values))
        if len(values) == 0:
            return best_feat, best_feat_val
        feat_num = len(values[0])  # 特征数量
        min_gini = self.gini_threshold  # 最小的基尼指数,初始化为最大值

        # 遍历每个特征
        for i in range(feat_num):
            unique_feat_vals = set([value[i] for value in values])  # 该特征所有特征取值
            if len(unique_feat_vals) == 1:
                continue

            # 统计该特征每个特征值划分后的基尼指数
            for val in unique_feat_vals:
                # 按val二分数据集
                values_equal, labels_equal, values_unequal, labels_unequal = self.split_data(values, labels, i,
                                                                                             val)
                # 每个数据集计算基尼指数
                gini_R1 = self.cal_gini(values_equal, labels_equal)
                gini_R2 = self.cal_gini(values_unequal, labels_unequal)

                # 加权求和
                gini = len(values_equal) / exam_num * gini_R1 + len(values_unequal) / exam_num * gini_R2
                if gini < min_gini:
                    min_gini = gini
                    best_feat = i
                    best_feat_val = val

        return best_feat, best_feat_val

    '''
    将数据集按某特征值切分成两个子集，并返回去掉该特征的两个子集
    :param values: 实例
    :param labels: 类别
    :param axis: 切分的特征(维度)
    :param axis_val: 切分的特征值
    :returns values_equal,labels_equal,values_unequal,labels_unequal: 切分并去掉切分特征后的实例、类别
    '''

    def split_data(self, values, labels, axis, axis_val):
        values_equal = []
        labels_equal = []
        values_unequal = []
        labels_unequal = []
        exam_num = len(values)
        for i in range(exam_num):
            val_vec = list(values[i])  # 特征向量
            # 去掉切分特征的向量
            sub_val = val_vec[:axis]
            sub_val.extend(val_vec[axis + 1:])  # 去掉切分特征
            if val_vec[axis] == axis_val:
                values_equal.append(sub_val)
                labels_equal.append(labels[i])
            else:
                values_unequal.append(sub_val)
                labels_unequal.append(labels[i])
        return values_equal, labels_equal, values_unequal, labels_unequal

    '''
    计算基尼指数
    :param values
    :param labels
    :return gini: 基尼指数
    '''

    def cal_gini(self, values, labels):
        exam_num = float(len(values))
        gini = 1.0

        # 统计每个类别的个数
        label_count = {}
        for label in labels:
            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1

        for key in label_count.keys():
            gini -= math.pow((label_count[key] / exam_num), 2)

        return gini

    '''
    决策树预测
    :param values: 需要预测的实例集合
    :return labels: 预测结果集合
    '''
    def predict(self,values):
        values = list(values)
        labels = []
        for value in values:
            labels.append(self.tree_search(value,self.root))
        return labels

    '''
    决策树搜索
    :param value: 搜索实例
    :param node: 当前搜索到的结点
    '''
    def tree_search(self,value,node):
        if 'is_leaf' in node.keys():
            return node['label']
        else:
            feat_index = self.features.index(node['split_feat'])
            if value[feat_index] == node['split_value']:
                return self.tree_search(value,node['left'])
            else:
                return self.tree_search(value, node['right'])




    '''
    决策树可视化入口
    :param filename: 输出文件的位置以及名字
    '''
    def tree_visualization(self,filename='file'):
        dot = Digraph(format='png')
        self.tree_visualization_core(self.root,dot)
        dot.render(filename=filename)

    '''
    决策树可视化核心代码
    使用每个字典的地址作为结点名字保证结点的唯一性
    :param node: 当前结点
    :param dot : 传递Digraph对象，创建结点用
    '''


    def tree_visualization_core(self, node,dot):
        if 'is_leaf' in node.keys():  # 单节点树
            dot.node(name=str(id(node)), label='类别: ' + node['label'], fontname="Microsoft YaHei", shape='square',color = '#00BFFF')
        else:
            dot.node(name=str(id(node)), label='切分特征: ' + node['split_feat'] + '\n切分值: ' + node['split_value'],
                     fontname="Microsoft YaHei", shape='square',color = '#00BFFF')
            # 左边
            self.tree_visualization_core(node['left'],dot)
            dot.edge(str(id(node)), str(id(node['left'])), 'yes')
            # 右边
            self.tree_visualization_core(node['right'], dot)
            dot.edge(str(id(node)), str(id(node['right'])), 'no')


if __name__ == '__main__':

    # 测试西瓜数据
    input_filename = './data/watermelon.csv'
    output_filename = './TreePng/watermelon'
    df = pd.DataFrame(pd.read_csv(input_filename, encoding='utf-8'))
    features = df.columns[:-1].values
    data = df.values
    values = data[:, :-1]
    labels = data[:, -1]

    '''交叉验证，70%训练集，30%测试集'''
    # 随机特征向量分成两组
    # 先将下标切分，后面方便将特征向量和标签切分
    train_index = list(range(len(values)))
    test_index = []
    test_len = int(len(values) * .3)  # 需要测试集的大小
    for i in range(test_len):
        rand = int(random.uniform(0, len(train_index)))
        test_index.append(train_index[rand])
        del train_index[rand]  # 将训练移入测试集的下标从训练下标删除

    train_values = values[train_index]
    train_labels = labels[train_index]
    test_values = values[test_index]
    test_labels = labels[test_index]



    # 训练决策树并画图
    cart = CART(train_values, train_labels, features)
    cart.tree_visualization(output_filename)

    # 计算结果
    correct_rate = 0.0
    correct_num = 0
    res_labels = cart.predict(test_values)
    for i in range(test_len):
        if test_labels[i] == res_labels[i]:
            correct_num += 1
    correct_rate = correct_num * 1.0 / test_len
    print('西瓜分类分类准确率为:{}', correct_rate)





