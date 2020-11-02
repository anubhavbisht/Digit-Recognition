# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
mnist = fetch_openml('mnist_784')
# print(mnist)
# print(mnist['data'])
# print(mnist['data'].shape)
# print(mnist['target'])
# print(mnist['target'].shape)
x, y = mnist['data'], mnist['target']
# print(x[0])
# print(x[0].shape)
# so first of all we want to convert 784*1 1d arry into 28*28 2d array
somedigit = x[3601]
#somedigitimage = somedigit.reshape(28, 28)
#plt.imshow(somedigitimage, cmap='binary', interpolation="nearest")
# plt.savefig('digit.png')
# plt.show()
# print(y[3601])
x_train, x_test = x[:6000], x[6000:7000]
y_train, y_test = y[:6000], y[6000:7000]
shuffleindex = np.random.permutation(6000)
x_train, y_train = x_train[shuffleindex], y_train[shuffleindex]
# creating a digit '2' detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train2 = (y_train == 2)  # creating a true false list
y_test2 = (y_test == 2)
# print(y_train2)
# print(y_test2)
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train2)
# print(clf.predict([somedigit]))
# cross validation of model
a = cross_val_score(clf, x_train, y_train2, cv=3, scoring="accuracy")
print(a)
print(a.mean())
# create a classifier which will classify a digit as not'2'
clf1 = LogisticRegression(tol=0.1)
y_trainnot2 = (y_train != 2)
y_testnot2 = (y_test != 2)
# print(y_trainnot2)
# print(y_testnot2)
clf1.fit(x_train, y_trainnot2)
b = cross_val_score(clf1, x_train, y_trainnot2, cv=3, scoring="accuracy")
print(b)
print(b.mean())
