import numpy as np
import matplotlib.pyplot as plt
x=np.array([3,2,1,101,99,98,18])
y=np.array([104,100,81,10,5,2,90])

colors = (['black','black','black','black','black','black','red'])
plt.scatter(x, y, c=colors)

plt.xlabel('打斗镜头数')
plt.ylabel("接吻镜头数")
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.show()