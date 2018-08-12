import decision_tree
import tree_plotter
import numpy as np

# 创建数据集
def createDataSet():
    dataSet = [['可以生存', '有', "鱼类"],
                ['可以生存', '有', "鱼类"],
                ['可以生存', '没有', "非鱼类"],
                ['不能生存', '有', "非鱼类"],
                ['不能生存', '有', "非鱼类"]]
    labels = ['不浮出水面是否可以生存', '是否有脚蹼']
    return dataSet, labels

if __name__ == "__main__":
    dataSet, labels = createDataSet()
    tree = decision_tree.createTree(dataSet, labels)
    print(tree)
    _, labels = createDataSet()
    result = decision_tree.classify(tree, labels, ["不能生存", "有"])
    print(result)


