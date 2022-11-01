# _*_ coding:utf-8 _*_
import re
import random

import numpy as np
from naive_bayes import NaiveBayes

'''读取数据'''
f = open('./data/email_data.txt', encoding='UTF-8')
email_list = []  # 存放每个邮件的分词
labels = []  # 存放每个邮件的分类
vocabulary_list = set([])  # 语料库
while True:
    # 处理一行，使用特殊字符分割字符串并转换成列表
    line = f.readline()
    if not line:
        break
    line =re.split('\W+', line)
    for i in range(len(line)):
        line[i] = line[i].lower()
    # 只保存标签正确的邮件
    if line[0] == 'ham' or line[0] == 'spam':
        email_list.append(line[1:])
        labels.append(line[0])
        vocabulary_list = vocabulary_list | set(line[1:])  # 更新语料库
vocabulary_list = list(vocabulary_list)  # 变回集合

'''对每封邮件根据语料库生成特征向量'''
values = []  # 特征向量
for i in range(len(email_list)):
    vec = [0] * len(vocabulary_list)  # 特征向量初始化为词料表大小
    for word in email_list[i]:
        if word in vocabulary_list:
            vec[vocabulary_list.index(word)] = 1
    values.append(vec)

values = np.array(values)
labels = np.array(labels)
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

'''训练并计算准确率'''
nb = NaiveBayes(train_values, train_labels)
test_len = len(test_values)
res_labels = []  # 返回集合
for i in range(test_len):
    res_labels.append(nb.predict(test_values[i]))


correct_rate = 0.0
correct_num = 0
for i in range(test_len):
    if test_labels[i] == res_labels[i]:
        correct_num += 1
correct_rate = correct_num * 1.0 / test_len
print('垃圾邮件分类准确率为:{}',correct_rate)
