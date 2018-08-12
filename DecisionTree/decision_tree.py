import math
import operator

# 计算香农熵
def calcShannonEnt(dataSet):
    # 获取数据集数据总个数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 获取数据的分类，数据集的最后一个数据即是该数据的分类
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算分类的概率
        prob = float(labelCounts[key]) / numEntries
        # 使用香农熵公式计算香农熵
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

# 划分数据集
# dataSet： 将被进行划分的数据集
# axis: 特征索引，表示按第几个特征进行划分，从0开始
# value: 特征的值，将把第axis个特征具有相同value的数据划分为同一个子集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉用于划分的特征，只保留还未用于划分的特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择划分数据集的最好特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 计算划分前的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 循环用每个特征对数据集进行划分
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 循环每个特征的所有取值，进行划分
        for value in uniqueVals:
            # 划分数据集，获得划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的数据比例
            prob = len(subDataSet) / float(len(dataSet))
            # 计算按特征i进行划分后，所有子集的总香农熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 比较信息增益，将信息增益大的特征赋予bestFeature变量
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 返回分类列表中出现次数最多的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet, labels):
    # 获取所有数据的分类，每条数据的最后一列为该数据的分类，因此可以通过索引-1获取。
    classList = [example[-1] for example in dataSet]
    # 如果分类列表中所有分类都相同，则直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果分类列表中分类不一样，且数据集中每条数据都只有一项数据了，
    # 这表示数据集中不包含特征数据了，也就是说所有特征都已经被用于划分数据集了。
    # 这种情况下，返回出现次数最多的分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 找出用于划分数据集的最好特征的索引
    bestFeature = chooseBestFeatureToSplit(dataSet)
    # 划分数据集的最好特征的名称
    bestFeatureLabel = labels[bestFeature]
    # 以最好特征名称为key，初始化决策树
    myTree = {bestFeatureLabel:{}}
    # 找到最好特征后，将其从特征列表中删除，以免后续重复使用特征进行分类。
    del(labels[bestFeature])
    # 获取最好特征所有的值
    featureValues = [example[bestFeature] for example in dataSet]
    # 去重
    uniqueVals = set(featureValues)
    # 循环特征的值
    for value in uniqueVals:
        subLabels = labels[:]
        # 用前面找出的最好特征与特征值划分数据集，再递归地对子集构建决策树
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

# 决策树分类器
# inputTree: 决策树
# featLabels: 特征标签列表
# testVec: 测试向量
def classify(inputTree, featLabels, testVec):
    # 获取决策树的第一个key，就是第一个分类特征
    firstStr = list(inputTree.keys())[0]
    # 获取第一个分类特征的值，也就是其分支
    secondDict = inputTree[firstStr]
    # 获取第一个特征的在特征标签列表中的索引
    featIndex = featLabels.index(firstStr)
    # 循环第一个特征的分支树
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # 如果分支是一个字典，说明还有包含子判断模块的子树
            if type(secondDict[key]).__name__ == 'dict':
                # 递归调用分类器，进入子树判断分类
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 否则说明已经找到最终分类，直接返回。
                classLabel = secondDict[key]
    return classLabel