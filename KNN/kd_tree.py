import queue

import numpy as np
import heapq
import pandas as pd

'''kd树的结点类'''

class KDNode:
    '''
    kd树构造函数
    :param dim: 结点分割的维度
    :param value: 当前结点对应实例
    :param label: 当前结点对应实例的类别
    :param left: 结点左孩子
    :param right: 结点右孩子
    :param dist: 当前结点到目标点的距离
    '''

    def __init__(self, dim, value, label, left, right):
        self.dim = dim
        self.value = value
        self.label = label
        self.left = left
        self.right = right
        self.dist = 1.7976931348623157e+308 # 初始化为最大值，这个不重要，会被覆盖的

    '''反着重写结点的比较函数，用于制造大根堆，因为heapq只能搞小根堆'''
    def __lt__(self, other):
        if self.dist>other.dist:
            return True
        else:
            return False




'''kd树'''


class KDTree:
    '''
    初始化参数并生成kd树
    其中实例和标签(类别)分别输入
    :param values: 实例
    :param labels: 类别
    '''

    def __init__(self, values, labels):
        self.values = np.array(values)
        self.labels = np.array(labels)
        self.dim_len = len(self.values[0]) if len(self.values) > 0 else 0  # 特征向量的维度，命名避免与k近邻的k混淆
        # 创建kd树
        self.root = self.create_KDTree(self.values, self.labels, 0)
        self.k = 0  # knn搜索个数
        self.knn_heap = []  # 临时存放knn结果的堆，注意这里默认是小顶堆，下面要用相反数

    '''
    递归创建kd树
    :param values: 实例
    :param labels: 类别
    :return: 该树的根节点
    '''

    def create_KDTree(self, values, labels, depth):
        if len(labels) == 0:
            return None

        dim = depth % self.dim_len  # 当前划分维度，注意这里不用+1，因为数组从0开始

        # 对实例和类别按实例某特征排序，不懂可参考 http://t.csdn.cn/8ZItF
        sort_index = values[:, dim].argsort()
        values = values[sort_index]
        labels = labels[sort_index]

        mid = len(labels) // 2  # 双除号向下取整
        node = KDNode(dim, values[mid], labels[mid], None, None)
        node.left = self.create_KDTree(values[0:mid], labels[0:mid], depth + 1)  # 递归创建左子树
        node.right = self.create_KDTree(values[mid + 1:], labels[mid + 1:], depth + 1)  # 递归创建右子树
        return node

    '''距离度量，这里使用欧氏距离'''

    def dist(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    """
    k近邻搜索的初始化
    主要作用是对搜索进行兜底
    :param target: 目标点
    :param k: 需要搜索近邻点的数量
    :return: 返回找到的实例和实例对应的标签组成的元组
    """

    def search_KNN(self, target, k):
        # 兜底
        if self.root is None:
            raise Exception('KD树不可为空')
        if k > len(self.values):
            raise ValueError('k值需小于等于实例数量')
        if len(target) != len(self.root.value):
            raise ValueError('目标点的维度和实例的维度大小需要一致')

        # 初始化并开始搜索
        self.k = k
        self.knn_heap = []
        self.search_KNN_core(self.root, target)
        res_values = []
        res_labels = []
        # 将结果转换一下
        for i in range(len(self.knn_heap)):
            res_values.append(self.knn_heap[i].value)
            res_labels.append(self.knn_heap[i].label)

        # print(res_labels)
        return (np.array(res_values),np.array(res_labels))

    '''
    k近邻搜索核心逻辑代码,由search_KNN调用
    :param root: 当前便利到的结点
    :param target: 目标点
    '''

    def search_KNN_core(self, node, target):
        if node is None:
            return []

        value = node.value
        dim = node.dim
        # 先往其中一个区域搜索
        if (target[dim] < value[dim]):
            ath_child = node.right  # 另一片区域对应结点
            if node.left is not None:
                self.search_KNN_core(node.left, target)
        else:
            ath_child = node.left  # 另一片区域对应结点
            if node.right is not None:
                self.search_KNN_core(node.right, target)

        # 处理本结点
        node.dist = self.dist(value, target)  # 结算本结点到目标节点的距离
        # 判断是否需要更新堆
        if len(self.knn_heap) < self.k:  # 堆没满直接进堆
            heapq.heappush(self.knn_heap, node)
        else:  # 堆若满则需要判断更新
            fathest_node_in_k = heapq.heappop(self.knn_heap)  # 已经找到的实例中距离目标点最远的实例
            if node.dist < fathest_node_in_k.dist:
                heapq.heappush(self.knn_heap, node)
            else:
                heapq.heappush(self.knn_heap, fathest_node_in_k)

        if ath_child is not None:
            fathest_node_in_k = heapq.heappop(self.knn_heap) # 获取堆顶供下面使用
            heapq.heappush(self.knn_heap,fathest_node_in_k)
            # 如果另一片区域与(以目标点为球心，以搜索到的集合中距离目标点最远的点到目标点的距离为半径)的超球体相交，则进入另一个子节点对应的区域搜索
            # 如果堆没满，也进入另一篇区域
            if len(self.knn_heap) < self.k or abs(ath_child.value[dim] - target[dim]) < fathest_node_in_k.dist:
                self.search_KNN_core(ath_child, target)

    '''先序输出，测试用'''
    def print_KDTree(self):
        stk = []
        p = self.root

        while len(stk) != 0 or p is not None:
            # 走到子树最左边
            while p is not None:
                stk.append(p)
                p = p.left

            if len(stk) != 0:
                cur_node = stk[len(stk) - 1]
                stk.pop()
                print(cur_node.value)
                # 若有则进入右子树，进行新一轮循环
                if cur_node.right is not None:
                    p = cur_node.right

# 一些测试的代码
# if __name__ == '__main__':
#     # 创建kd树
#     df = pd.DataFrame(pd.read_excel('./data/knn.xlsx'))
#     values = df.values[:, :-1]
#     labels = df.values[:, -1]
#     kdtree = KDTree(values, labels)
#     # kdtree.print_KDTree()
#     kdtree.search_KNN([2.1,4.7], 5)
