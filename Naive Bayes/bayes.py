# -*- coding: UTF-8 -*-
import numpy as np
from functools import reduce

"""
函数说明:创建实验样本

Parameters:
	无
Returns:
	postingList - 实验样本切分的词条
	classVec - 类别标签向量
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""
def loadDataSet():
	# 创建一个示例邮件矩阵，每一行为一封邮件
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	# 邮件的是否侮辱的类别标签，其中0代表非侮辱性，1代表侮辱性
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec


"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""


def createVocabList(dataSet):
	# 初始化一个空的不重复的列表用于记录所有邮件中出现的词条
	vocabSet = set([])
	for document in dataSet:				
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""


def setOfWords2Vec(vocabList, inputSet):
	# 初始化一个长度为词汇表长度的全零向量
	returnVec = [0] * len(vocabList)
	# 对于每封邮件，将其中存在的词条对应的向量位置置为1，进行编码
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary!" % word)
	# 返回邮件编码向量
	return returnVec


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 非的条件概率数组
	p1Vect - 侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""
def trainNB0(trainMatrix,trainCategory):
	# 训练矩阵为矩阵化的文档矩阵，每一行为one-hot编码后的向量，列数为文档个数
	numTrainDocs = len(trainMatrix)
	# 记录每篇文档编码向量的长度，即为vocabList的长度
	numWords = len(trainMatrix[0])
	# 由于侮辱类用字母1表示，故sum(trainCategory)可以用于表示训练集中侮辱类的个数，pAbusive表示侮辱类的概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	# 初始化两个长度为编码向量长度的全零数组分别用于记录非侮辱类和侮辱类下各个词出现的条件概率p(wi|0)
	p0Num = np.zeros(numWords)
	p1Num = np.zeros(numWords)
	# 初始化p0Deom和P1Denom，用于记录非侮辱类和侮辱类下出现的词的总个数
	p0Denom = 0.0
	p1Denom = 0.0
	# 循环判断
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			# 将trainMatrix的第i行加入p1Num中
			p1Num += trainMatrix[i]
			# 累加已经循环过的非侮辱类文档的词条数
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# 计算出非侮辱类和侮辱类下各个词出现的条件概率p(wi|0)
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect, p1Vect, pAbusive

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 非侮辱类的条件概率数组
	p1Vec -侮辱类的条件概率数组
	pClass1 - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	# 将待分类的词条向量与侮辱性的词条条件概率向量相乘，再通过reduce函数将结果向量的各个值相乘得到p(w|1)，再乘以pClass1得到p(w|1) * p(1)，即似然函数乘以先验概率
	p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1
	p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
	print('p0:', p0)
	print('p1:', p1)
	if p1 > p0:
		return 1
	else: 
		return 0

"""
函数说明:测试朴素贝叶斯分类器

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""


def testingNB():
	# 创建实验样本
	listOPosts, listClasses = loadDataSet()
	# 创建词汇表
	myVocabList = createVocabList(listOPosts)
	# 初始化训练邮件矩阵
	trainMat = []
	# postinDoc代表一封邮件，通过对每封邮件向量化并且通过append函数生成编码矩阵
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	# 训练朴素贝叶斯分类器
	p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
	# 测试样本1
	testEntry = ['love', 'my', 'dalmation']
	# 对测试样本进行编码并向量化
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	# 注意输出的p1和p0都为0
	if classifyNB(thisDoc, p0V, p1V, pAb):
		print(testEntry, '属于侮辱类')
	else:
		print(testEntry, '属于非侮辱类')
	# 测试样本2
	testEntry = ['stupid', 'garbage']
	# 对测试样本进行编码并向量化
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	if classifyNB(thisDoc, p0V, p1V, pAb):
		print(testEntry, '属于侮辱类')
	else:
		print(testEntry, '属于非侮辱类')


if __name__ == '__main__':
	testingNB()
