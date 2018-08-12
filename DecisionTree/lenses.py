import decision_tree

if __name__ == '__main__':
    fr = open('lenses.txt')
    # 读取数据文件的每一行，然后以\t分割成列表
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
    # 使用decision_tree实现的createTree()函数创建决策树
    lensesTree = decision_tree.createTree(lenses, lensesLabels)
    print(lensesTree)
    
    # 注意，我们在使用分类器时，要重新传入分类标签列表，不能重用前面的分类标签列表。因为在创建决策树函数中，会删除标签列表里的数据。
    labels = ["age", "prescript", "astigmatic", "tearRate"]
    # 使用分类器函数预测未知数据
    result = decision_tree.classify(lensesTree, labels, ["young", "hyper", "yes", "reduced"])
    print(result)