# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:梯度上升算法测试函数

求函数f(x) = -x^2 + 4x的极大值

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""


def Gradient_Ascent_test():
	# 定义f(x) = - x^2 + 4 * x的导数
	def f_prime(x_old):
		return -2 * x_old + 4
	# 给自变量一个初始值
	x_old = -1
	# 梯度上升算法初始值，即从x = 0时开始
	x_new = 0
	# 步长，也就是学习率，控制更新的幅度
	alpha = 0.01
	# 设定函数停止的依据:当前后两个函数值之差小于precision时，函数停止
	precision = 0.00000001
	while abs(x_new - x_old) > precision:
		x_old = x_new
		# 函数迭代式，每一次的增量为 alpha * f_prime(x_old)
		x_new = x_old + alpha * f_prime(x_old)
	# 打印最终求解的极值近似值
	print(x_new)


"""
函数说明:加载数据

Parameters:
	无
Returns:
	dataMat - 数据列表
	labelMat - 标签列表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""


def loadDataSet():
	# 初始化空的数据矩阵
	dataMat = []
	# 初始化空的标签向量
	labelMat = []
	# 打开文件
	fr = open('testSet.txt')
	# 对文件逐行读取，并且拆分为数据矩阵和标签向量
	for line in fr.readlines():
		# 通过strip函数去除每一行行尾的回车，通过split函数将每一行划分为词向量
		lineArr = line.strip().split()
		# 生成数据矩阵：textSet.txt中每一行的前两个分量为两个特征，取x1为lineArr[0]，x2为lineArr[1]，偏差bias设定为1
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		# 生成标签向量：textSet.txt每一行的最后一个分量为标签值（0或者1）
		labelMat.append(int(lineArr[2]))
	# 关闭文件
	fr.close()
	# 返回数据矩阵，标签向量
	return dataMat, labelMat


"""
函数说明:sigmoid函数

Parameters:
	inX - 数据
Returns:
	sigmoid函数
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""


def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法

Parameters:
	dataMatIn - 数据集
	classLabels - 数据标签
Returns:
	weights.getA() - 求得的权重数组(最优参数)
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""


def gradAscent(dataMatIn, classLabels):
	# 将dataMatIn转化为numpy的mat
	dataMatrix = np.mat(dataMatIn)
	# 将classLabels转换成numpy的mat，并进行转置
	labelMat = np.mat(classLabels).transpose()
	# dataMatrix的行数和列数为m，n
	m, n = np.shape(dataMatrix)
	# 步长，又称学习率
	alpha = 0.001
	# 最大迭代次数
	maxCycles = 500
	# 初始化一个长度为n的全1权重向量
	weights = np.ones((n, 1))

	for k in range(maxCycles):
		# 梯度上升矢量化公式
		h = sigmoid(dataMatrix * weights)
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	# 将矩阵转化为数组
	return weights.getA()


"""
函数说明:绘制数据集

Parameters:
	weights - 权重参数数组
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-30
"""


def plotBestFit(weights):
	# 加载数据集
	dataMat, labelMat = loadDataSet()
	dataArr = np.array(dataMat)
	# 记录样本点的个数
	n = np.shape(dataMat)[0]
	# 正样本
	xcord1 = []
	ycord1 = []
	# 负样本
	xcord2 = []
	ycord2 = []
	for i in range(n):
		# 根据数据集标签进行分类
		if int(labelMat[i]) == 1:
			# 1为正样本
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			# 0为负样本
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	# 绘图
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# 绘制正样本
	ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
	# 绘制负样本
	ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.title('Best Fit')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()		


if __name__ == '__main__':
	dataMat, labelMat = loadDataSet()	
	weights = gradAscent(dataMat, labelMat)
	plotBestFit(weights)
