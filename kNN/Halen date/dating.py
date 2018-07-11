from numpy import *
import operator
import matplotlib.pyplot as plt

def classifier(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

def showDataSet(dataSet):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(15,10))  
    
    axs[0][0].scatter(dataSet[:, 0], dataSet[:, 1], 15.0*array(labels), 15.0*array(labels))
    axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间百分比')
    axs[0][0].set_xlabel('每年获得的飞行常客里程数')
    axs[0][0].set_ylabel('玩视频游戏所消耗时间百分比')
    
    axs[0][1].scatter(dataSet[:, 0], dataSet[:, 2], 15.0*array(labels), 15.0*array(labels))
    axs[0][1].set_title('每年获得的飞行常客里程数与每周消费的冰激淋公升数')
    axs[0][1].set_xlabel('每年获得的飞行常客里程数')
    axs[0][1].set_ylabel('每周消费的冰激淋公升数')
    
    axs[1][0].scatter(dataSet[:, 1], dataSet[:, 2], 15.0*array(labels), 15.0*array(labels))
    axs[1][0].set_title('玩视频游戏所消耗时间百分比与每周消费的冰激淋公升数')
    axs[1][0].set_xlabel('玩视频游戏所消耗时间百分比')
    axs[1][0].set_ylabel('每周消费的冰激淋公升数')
    
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    testRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * testRatio)
    trainDataMat = normMat[numTestVecs:m,:]
    trainLabels = datingLabels[numTestVecs:m]
    k = 3
    errorCount = 0.0
    for i in range(numTestVecs):
        testIdata = normMat[i, :]
        classResult = classifier(testIdata, trainDataMat, trainLabels, k)
        print("分类器预测结果是: ", classResult, " 实际结果是: ", datingLabels[i])
        if (classResult != datingLabels[i]):
            errorCount += 1.0
    print("总的错误率是: ", errorCount / float(numTestVecs))

def classifyPerson():
    resultList = ['你应该不喜欢他', '你可能会有一点喜欢他', '他应该是你喜欢的人']
    percentTats = float(input("玩视频游戏所消耗时间百分比？"))
    ffMiles = float(input("每年获得的飞行常客里程数？"))
    iceCream = float(input("每周消费的冰淇淋公升数？"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    preProcessInArr = (inArr - minVals) / ranges
    classifierResult = classifier(preProcessInArr, normMat, datingLabels, 3)
    print(">> 预测结果：", resultList[classifierResult - 1])

if __name__ == '__main__':
    dataSet, labels = file2matrix('datingTestSet.txt')
    # showDataSet(dataSet)
    # datingClassTest()
    classifyPerson()