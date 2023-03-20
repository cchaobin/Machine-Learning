# -*- coding: UTF-8 -*-
import numpy as np
import random
import re


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
    # 创建一个空的不重复列表，通过set不重复的性质去重
    vocabSet = set([])
    for document in dataSet:
        # 对邮件集中的所有邮件向量取并集
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
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


"""
函数说明:根据vocabList词汇表，构建词袋模型

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词袋模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-14
"""


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 非侮辱类的条件概率数组
	p1Vect - 侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""


def trainNB0(trainMatrix, trainCategory):
    # 训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 文档编码向量的长度
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为文档编码向量的长度,拉普拉斯平滑（Laplace Smoothing）
    p0Denom = numWords
    p1Denom = numWords
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 打印非侮辱类和侮辱类下所有的条件概率的和是否为1
    print('非侮辱类的所有条件概率之和为%s' % sum(p0Num / p0Denom))
    print('侮辱类的所有条件概率之和为%s' % sum(p1Num / p1Denom))
    # 通过利用对数函数在趋于0时的大导数特性，对趋于0的概率取对数，防止下溢出
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
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
    # 对应元素相乘，logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
函数说明:接收一个大字符串并将其解析为字符串列表

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-14
"""


def textParse(bigString):
    # 通过正则split会将字符串分割成单个字符【python3.5+】; 这里使用\W 或者\W+ 都可以将字符数字串分割开，产生的空字符将会在后面的列表推导式中过滤掉
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    listOfTokens = re.split(r'\W+', bigString)
    # 用于避免将大小写不同的相同词识别成为不同的词，同时将2个字母以下的词汇剔除
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

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
    2017-08-14
"""
def spamTest():
    docList = []
    classList = []
    fullText = []
    # 遍历25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        # 通过不断append邮件向量，生成邮件矩阵（未编码）
        docList.append(wordList)
        fullText.append(wordList)
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记非垃圾邮件，1表示垃圾文件
        classList.append(0)
        # 创建词汇表，不重复
    vocabList = createVocabList(docList)

    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    trainingSet = list(range(50))
    testSet = []
    # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
    for i in range(10):
        # 随机选取索索引值
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        # 创建训练集矩阵和训练集类别标签系向量
    trainMat = []
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签系向量中
        trainClasses.append(classList[docIndex])
        # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 测试集的词集模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 如果分类错误
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 错误计数加1
            errorCount += 1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
