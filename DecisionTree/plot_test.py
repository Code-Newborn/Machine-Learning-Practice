import tree_plotter
import matplotlib.patches as mpatch

def retrieveTree(i):
    listOfTrees = [{'不浮出水面是否可以生存': {'不能生存': '非鱼类', '可以生存': {'是否有脚蹼': {'没有': '非鱼类', '有': '鱼类'}}}},
                   {'不浮出水面是否可以生存': {'不能生存': '非鱼类', '可以生存': {'是否有脚蹼': {'没有': {'是否有头': {'没有': '非鱼类', '有': '鱼类'}}, '有': '非鱼类'}}}},
                   {'是否有脚蹼': {'没有': '非鱼类', '有': {'不浮出水面是否可以生存': {'可以生存': '鱼类', '不能生存': '非鱼类'}}}}]

    return listOfTrees[i]

if __name__ == '__main__':
    myTree = retrieveTree(1)
    tree_plotter.createPlot(myTree)
