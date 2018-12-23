import numpy as np

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to' ,'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'hime'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'bying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

# 训练算法
# trainMatrix: 训练数据，也就是前面得到的词向量
# trainCategory: 训练数据的标签
def train(trainMatrix, trainCategory):
    # 训练数据中包含的文档个数
    numTrainDocs = len(trainMatrix)
    # 每条训练数据中单词个数，等于词汇表的长度
    numWords = len(trainMatrix[0])
    # 训练数据中类别1的概率，也就是垃圾邮件的概率
    # 这里比较巧妙的使用了sum()函数得到类别1的文档个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建两个向量，大小均为词汇表长度
    # p0Num向量表示为类别0的所有文档各个词出现的总次数
    p0Num = np.ones(numWords)
    # p1Num向量表示类别1的所有文档各个词出现的总次数
    p1Num = np.ones(numWords)
    # p0Denom为类别0的所有文档的所有词总数
    p0Denom = 2.0
    # p1Denom为类别1的所有文档的所有词总数
    p1Denom = 2.0
    # 循环所有文档，累计p0Num，p1Num，p0Denom，p1Denom
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算各个类别的条件概率
    # p1Vect为类别1的条件概率
    p1Vect = np.log(p1Num/p1Denom)
    # p0Vect为类别0的条件概率
    p0Vect = np.log(p0Num/p0Denom)
    # 返回三个概率
    return p0Vect, p1Vect, pAbusive
    
# 分类器函数
# vec2Classify 待预测的词向量
# p0Vec，p1Vec, pClass1分别为train()函数训练输出的三个概率值
def classify(vec2Classify, p0Vec, p1Vec, pClass1):
    # 待预测向量为类别1的概率，这里利用了前面提到的对数公式
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    # 待预测向量为类别0的概率，这里利用了前面提到的对数公式
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    # 如果p1大于p0，这预测结果为类别1，否则为类别0
    if p1 > p0:
        return 1
    else:
        return 0

def testing():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postingDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, pAb = train(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classify(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classify(thisDoc, p0V, p1V, pAb))

testing()