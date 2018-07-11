from numpy import *
import operator
import sys

def classifier(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    dataSet = array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    labels = ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片']
    return dataSet, labels

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    input = [int(sys.argv[1]), int(sys.argv[2])]
    k = 4
    output = classifier(input, dataSet, labels, k)
    print(output)
