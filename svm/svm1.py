# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics

################
# initial data #
################

digits = datasets.load_digits()


#####################
# flatten the image #
#####################
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



clfs = [
        ('SVM with linear kernel', svm.SVC(kernel='linear', gamma=0.001)),
        ('SVM with rbf kernel', svm.SVC(kernel='rbf', gamma=0.001))
       ]


for i in range(2):
    images_and_labels = list(zip(digits.images, digits.target))
    plt.figure(num=clfs[i][0])
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    clf = clfs[i][1]
    clf.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

    expected = digits.target[n_samples / 2:]
    predicted = clf.predict(data[n_samples / 2:])


    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


    images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

plt.show()

