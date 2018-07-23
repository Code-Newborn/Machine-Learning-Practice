import numpy as np
import operator
import os

realpath = os.path.realpath(__file__)
dir = os.path.dirname(realpath)
trainingDir = os.path.join(dir, 'trainingDigits')
testDir = os.path.join(dir, 'testDigits')

def img2vector(filename):
    vector = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(line[j])
    return vector

def classifier(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
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

def handwritingClassTest():
    labels = []
    trainingFileList = os.listdir(trainingDir)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileName = trainingFileList[i]
        fileStr = fileName.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        labels.append(classNumber)
        trainingMat[i, :] = img2vector(os.path.join(trainingDir, fileName))
    testFileList = os.listdir(testDir)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        fileStr = fileName.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(os.path.join(testDir, fileName))
        result = classifier(vectorUnderTest, trainingMat, labels, 3)
        print("分类器预测结果: %d, 实际答案是: %d" % (result, classNumber))
        if result != classNumber: 
            errorCount += 1.0
    print("\n预测错误数: %d" % errorCount)
    print("\n错误率为: %f" % (errorCount/float(mTest)))

handwritingClassTest()


