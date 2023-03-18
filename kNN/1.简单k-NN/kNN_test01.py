# -*- coding: utf-8 -*-
import numpy as np
import operator
import collections


"""
函数说明:创建数据集

Parameters:
	无
Returns:
	group - 数据集
	labels - 分类标签
Modify:
	2017-07-13
"""


def createDataSet():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['Love', 'Love', 'Action', 'Action']
    return group, labels


"""
函数说明:kNN算法(K Nearest-Neighbor Algorithm),分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2017-11-09 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Use list comprehension and Counter to simplify code
	2017-07-13
"""


def classify0(inx, dataset, labels, k):
    # diff = np.sum((inx - dataset) ** 2, axis=1)
    # inx为测试点，下式计算测试点和训练集合中的每个测试点的距离之和
    dist = np.sum((inx - dataset) ** 2, axis=1) ** 0.5
    # 通过dist.argsort()将距离测试点最近的k个点取出，利用list comprehension将k个距离最近点的标签存入k_labels中
    k_labels = [labels[index] for index in dist.argsort()[0: k]]
    # 通过引入collection库，将出现次数最多的标签取出，其中collection.Counter(k_labels).most_common(1)会输出('Action',2),[0][0]将label拿到
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


if __name__ == '__main__':
    # 创建带标签的训练数据集
    group, labels = createDataSet()
    # 测试集/需要预测的点
    test = [101, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)
