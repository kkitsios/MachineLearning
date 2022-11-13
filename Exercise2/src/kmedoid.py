import numpy as np
from math import log2
from sklearn.metrics import confusion_matrix
import idx2numpy as idx2numpy
import Histogram, ReduceDataSet


class k_medoids:
    def __init__(self, k=10):  # total number of clusters=10
        self.k = k

    def fit(self, data):

        # For 1st clustering
        self.medoids = []
        self.cost = 0
        self.clusters = []
        # rand = np.random.choice([i for i in range(len(data))],self.k, replace=False)
        # self.medoids = [data[i] for i in rand]
        # print(self.medoids)
        # exit(0)
        for i in range(self.k):
            self.medoids.append(data[i])  # selecting k random points out of the dataset as the medoids (1st step)
        for j in range(self.k):
            self.clusters.append([])

        for point in data:
            # list containing distances of each point in the dataset from the medoids and finding minimum distance
            # sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
            # (abs((point-m)).sum()) for m in self.medoids

            distance = [np.sum(np.where(point != 0, point * np.log(point / m), 0)) for m in self.medoids]
            min_distance = min(distance)
            # calculating the cost (cost is the distance of each point from its medoid i.e., total of min distances) 
            self.cost += min_distance
            o = distance.index(min_distance)
            # clustering (i.e., associating each point to the closest medoid)
            self.clusters[o].append(point)

        # now, at each iteration we are finding the replacement of our medoids
        # we will stop on reaching the max no. of iteration or if there is no changes in the cost
        while (True):
          #  print("While loop")
            # lists that will store new clusters, new cost and the new medoids 
            new_clusters = []
            self.new_cost = 0
            self.new_medoids = []
            for j in range(self.k):
                new_clusters.append([])

            for j in range(self.k):
                # list for storing the total distance of a particular point from all other points in the same cluster
                dist_with_each_point_in_same_cluster = []
                l = []  # list for storing the distance of each and every point from all other points of the same cluster
               # print("clusters",len(self.clusters[j]))
                for point in self.clusters[j]:
                    dist_with_each_point_in_same_cluster = sum([np.sum(np.where(point != 0, point * np.log(point / d), 0)) for d in self.clusters[j]])
                    l.append(dist_with_each_point_in_same_cluster)
                #print("j=",j)
                minima = min(l)  # finding the minimum distance
                q = l.index(minima)
                # point with the min distance from all the points in the same cluster is taken as the new medoid
                self.new_medoids.append(self.clusters[j][q])
           # print("Fin 1st loop")

                # now, finding the new clustering and the new cost
            for point in data:
                # list containing distances of each point in the dataset from the new medoids and finding minimum distance
                distance = [np.sum(np.where(point != 0, point * np.log(point / m), 0)) for m in self.new_medoids]
                min_distance = min(distance)
                # calculating the new cost
                self.new_cost += min_distance
                o = distance.index(min_distance)
                # new clustering
                new_clusters[o].append(point)
           # print("Fin 2st loop")
            # if the cost decreases with the new medoids we will replace the old medoids, old clusters and the old cost with the new one
            # if the cost increases with the new medoids we will not replace the old medoids, old clusters and the oldcost with the new one
            # print(abs(self.new_cost - self.cost))
            if self.new_cost < self.cost:
                self.medoids = self.new_medoids
                self.clusters = new_clusters
                self.cost = self.new_cost

            # if the cost remains same we will stop and come out of the iterations

            elif abs(self.new_cost - self.cost) <= 0.01:
                break

    # final clustering according to the new medoids
    def predict(self, test_data):
        pred = []
        for point in test_data:
            # finding distance of each point from the final medoids and get the min distance
            distance = [np.sum(np.where(point != 0, point * np.log(point / m), 0)) for m in self.medoids]
            minDistance = min(distance)
            l = distance.index(minDistance)
            pred.append(l)  # associating each point to the closest medoid
        return pred

    def purity(self, y_true, y_pred):
        np.seterr(divide='ignore', invalid='ignore')
        cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    def f_measure(self, y_true, y_pred):
        np.seterr(divide='ignore', invalid='ignore')
        cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tn = cm.sum() - (tp + fn + fp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * ((precision * recall) / (precision + recall))
        f1[np.isnan(f1)] = 0
        return f1.sum()


trainImagePath = "../dataset/train-images-idx3-ubyte"
trainLabelPath = "../dataset/train-labels-idx1-ubyte"
testImagePath = "../dataset/t10k-images-idx3-ubyte"
testLabelPath = "../dataset/t10k-labels-idx1-ubyte"

x_train = idx2numpy.convert_from_file(trainImagePath)
x_test = idx2numpy.convert_from_file(testImagePath)

y_train = idx2numpy.convert_from_file(trainLabelPath)
y_test = idx2numpy.convert_from_file(testLabelPath)

x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

x_train,y_train = ReduceDataSet.reduce(x_train,y_train,10,6000)
bins = 16
histArray = Histogram.histogram(x_train,bins)
testhistArray=Histogram.histogram(x_test,bins)


print ("-----------------------Kmedoid clustering algorithm---------------------------------")
print("########################R2-Data Represantation###########################")
model = k_medoids()
model.fit(histArray)
clusters = model.predict(testhistArray)

print("Purity score {:.3f}%".format(model.purity(y_true=y_test, y_pred=clusters)*100))
print("F-measure score {:.3f}%".format(model.f_measure(y_true=y_test, y_pred=clusters)*100))

