import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm, metrics
import sklearn
from Resize import resize
from math import pow
trainImagePath = "../dataset/train-images-idx3-ubyte"
trainLabelPath = "../dataset/train-labels-idx1-ubyte"
testImagePath = "../dataset/t10k-images-idx3-ubyte"
testLabelPath="../dataset/t10k-labels-idx1-ubyte"

x_test = idx2numpy.convert_from_file(trainImagePath)
y_test = idx2numpy.convert_from_file(trainLabelPath)

x_train = idx2numpy.convert_from_file(testImagePath)
y_train = idx2numpy.convert_from_file(testLabelPath)

x_train=resize(x_train,28,28,7,7)
x_test=resize(x_test,28,28,7,7)



x_train=x_train.reshape((x_train.shape[0], 7*7))
x_test=x_test.reshape((x_test.shape[0], 7*7))

x_train=x_train/255.0
x_test=x_test/255.0

linear=svm.SVC(kernel='linear',C=1,decision_function_shape='ovr').fit(x_train,y_train)
sigmoid=svm.SVC(kernel='sigmoid',C=1,decision_function_shape='ovr').fit(x_train,y_train)
x_train = x_train.astype('float32')
cosine=svm.SVC(kernel=sklearn.metrics.pairwise.cosine_similarity,C=1,decision_function_shape='ovr').fit(x_train,y_train)

lpred=linear.predict(x_test)
sigpred=sigmoid.predict(x_test)
cospred=cosine.predict(x_test)


print("Linear Kernel")
print("acc={:.3f}%".format(metrics.accuracy_score(y_test, lpred)*100.0))
print("f1={:.3f}%\n".format(metrics.f1_score(y_true=y_test, y_pred=lpred, average='macro')*100.0))
print("Sigmoid Kernel")
print("acc={:.3f}%".format(metrics.accuracy_score(y_test, sigpred)*100.0))
print("f1={:.3f}%\n".format(metrics.f1_score(y_true=y_test, y_pred=sigpred, average='macro')*100.0))
print("Cosine Kernel")
print("acc={:.3f}%".format(metrics.accuracy_score(y_test, cospred)*100.0))
print("f1={:.3f}%".format(metrics.f1_score(y_true=y_test, y_pred=cospred, average='macro')*100.0))
