# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, linear_model, svm, datasets


# 提取训练集，其中特征个数为2，目标变量种类为2
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
# 构造预测集
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

# 分类器们
clfs = [
    ('Decision Tree', tree.DecisionTreeClassifier()),
    ('Logistic Regression', linear_model.LogisticRegression()),
    ('SVM with linear kernel', svm.SVC(kernel='linear')),
    ('SVM with rbf kernel', svm.SVC(kernel='rbf'))
]

# 绘图
for i in range(len(clfs)):
    clf = clfs[i][1]
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(clfs[i][0])

plt.show()
