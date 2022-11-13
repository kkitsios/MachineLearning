import idx2numpy as idx2numpy
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Resize import resize


trainImagePath = "../dataset/train-images-idx3-ubyte"
trainLabelPath = "../dataset/train-labels-idx1-ubyte"
testImagePath = "../dataset/t10k-images-idx3-ubyte"
testLabelPath="../dataset/t10k-labels-idx1-ubyte"

x_train = idx2numpy.convert_from_file(trainImagePath)
y_train = idx2numpy.convert_from_file(trainLabelPath)

x_test = idx2numpy.convert_from_file(testImagePath)
y_test = idx2numpy.convert_from_file(testLabelPath)

x_train=resize(x_train,28,28,7,7)
x_test=resize(x_test,28,28,7,7)

x_train=x_train.reshape((x_train.shape[0], 7*7))
x_test=x_test.reshape((x_test.shape[0], 7*7))

x_train=x_train/255.0
x_test=x_test/255.0

k=[1,5,10]
accuracy=[]
F1score=[]
for i in k:
     knn=KNeighborsClassifier(n_neighbors=i, metric="cosine")
     knn.fit(x_train,y_train)
     y_pred=knn.predict(x_test)
     accuracy.append(metrics.accuracy_score(y_test, y_pred))
     F1score.append(metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro'))

print(15*'-'+'cosine distance'+15*'-')
for i in range(3):
     print("k={}: [acc={:.3f}%]".format(k[i],accuracy[i]*100.0), end="  ")
     print()
     print("k={}: [f1 score={:.3f}%]".format(k[i],F1score[i]*100.0), end="  ")
     if i!=2: print("\n-----------------------")
     else: print()

accuracy=[]
F1score=[]
for i in k:
     knn=KNeighborsClassifier(n_neighbors=i,metric='minkowski', p=2)
     knn.fit(x_train,y_train)
     y_pred=knn.predict(x_test)
     accuracy.append(metrics.accuracy_score(y_test, y_pred))
     F1score.append(metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro'))
print(15*'-'+'euclidean distance'+15*'-')
for i in range(3):
     print("k={}: [acc={:.3f}%]".format(k[i],accuracy[i]*100.0), end="  ")
     print()
     print("k={}: [f1 score={:.3f}%]".format(k[i],F1score[i]*100.0), end="  ")
     if i!=2: print("\n-----------------------")
     else: print()
#acc=1: 82.95 5: 84.17 10: 84.682
#f1=1: 83.013 5: 84.03 10: 84.027
#acc=1: 84.07 5: 85.370 10: 84.8
#f1=1:84.072 5: 85.223 10: 84.682
#######################
