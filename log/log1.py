# -*- coding: utf-8 -*-

from numpy import *
from sklearn import linear_model, metrics


#############
# train data#
#############

ftrain = open('horse/horseColicTraining.txt')
train_digits = array(ftrain.read().split())
n = len(train_digits)
train_data = train_digits.reshape((n/22, 22))
train_X = train_data[:, :21].astype(float)
train_Y = train_data[:, 21].astype(float)


##############
# classifier #
##############

clf = linear_model.LogisticRegression()
clf.fit(train_X, train_Y)


#############
# test data #
#############

ftest = open('horse/horseColicTest.txt')
test_digits = array(ftest.read().split())
m = len(test_digits)
test_data = test_digits.reshape((m/22, 22))
test_X = test_data[:, :21].astype(float)
test_Y = test_data[:, 21].astype(float)


predicted = clf.predict(test_X)


print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(test_Y, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_Y, predicted))


count = 0
for i in range(len(test_Y)):
    if predicted[i] != test_Y[i]:
        count += 1

print 'error_rate is %f' % (float(count)/len(test_Y))






