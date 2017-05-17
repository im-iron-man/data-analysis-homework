import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree, datasets


iris = datasets.load_iris()
X = iris.data
Y = iris.target
h = 0.02


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



x_min = [0,0,0,0]
x_max = [0,0,0,0]
x_min[0], x_max[0] = X[:, 0].min() - 1, X[:, 0].max() + 1
x_min[1], x_max[1] = X[:, 1].min() - 1, X[:, 1].max() + 1
x_min[2], x_max[2] = X[:, 2].min() - 1, X[:, 2].max() + 1
x_min[3], x_max[3] = X[:, 3].min() - 1, X[:, 3].max() + 1


fig = plt.figure()
fig, axes = plt.subplots(4,4)



for i in range(4):
    for j in range(4):
        clf = tree.DecisionTreeClassifier()
        clf.fit(X[:, [i,j]], Y)
        xx, yy = np.meshgrid(np.arange(x_min[i], x_max[i], h), np.arange(x_min[j], x_max[j], h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


        ZZ = Z.reshape(xx.shape)

        axes[i, j].pcolormesh(xx, yy, ZZ, cmap=cmap_light)

        axes[i, j].scatter(X[:, i], X[:, j], c=Y, cmap=cmap_bold)
        axes[i, j].set_xlim(xx.min(), xx.max())
        axes[i, j].set_ylim(yy.min(), yy.max())

plt.show()