import math
from sklearn.metrics import confusion_matrix
import numpy as np
import idx2numpy as idx2numpy
import Histogram, ReduceDataSet


class Hierarchical:   
	def __init__(self,data): 
		'''
		constructor of the class, it takes the main data frame as input
		'''
		self.data = data  
		self.n_samples, self.n_features = data.shape

	def DistanceMatrix(self,data,distance):
		'''
		arguement
		---------
		data - the dataset whose Similarity matrix we are going to calculate
		returns
		-------
		the distance matrix.
		'''
		N = len(data)
		if (distance == "Cosine"):
			similarity_mat = np.ones([N, N]) #for cosine np.ones
		else:
			similarity_mat = np.zeros([N, N]) #for cosine np.ones
		for i in range(N):
			for j in range(N):
				similarity_mat[i][j]=self.SimilarityMeasure(data[i],data[j],distance)
		return similarity_mat

	def SimilarityMeasure(self,data1, data2, distance):
		'''
		arguements
		----------
		data1, data2 - vectors, between which we are going to calculate similarity
		returns
		-------
		distance between the two vectors
		'''
		N=self.n_features
		if distance=='Euclidean':
			# for L2 norm or pythagorean distance
			dist = 0
			for i in range(N):
				dist += (data1[i] - data2[i])**2

			dist = math.sqrt(dist)
			return dist

		if distance=='Cosine':
			# form cosine similarity = a.b/|a|.|b|
			dot_prod = 0
			data1_mod = np.linalg.norm(data1)
			data2_mod = np.linalg.norm(data2)
			for x in range(N):
				dot_prod += data1[x]*data2[x]
			return (1-dot_prod/(data1_mod*data2_mod))

		if distance=='Manhattan':
			# for L1 norm or pythagorean distance
			dist = 0
			for i in range(N):
				dist += abs(data1[i] - data2[i] )
			return dist

	def fit(self,k,distance):
		'''
		this method uses the main Divisive Analysis algorithm to do the clustering
		arguements
		----------
		k - integer
					 number of clusters we want
		
		returns
		-------
		cluster_labels - numpy array
						 an array where cluster number of a sample corrosponding to 
						 the same index is stored
		'''
		similarity_matrix = self.DistanceMatrix(self.data,distance) # similarity matrix of the data
		clusters = [list(range(self.n_samples))]      # list of clusters, initially the whole dataset is a single cluster
		while True:
			c_diameters = [np.max(similarity_matrix[cluster][:, cluster]) if (len(similarity_matrix[cluster][:, cluster]) != 0) else -1 for cluster in clusters]  #cluster diameters
			# c_diameters = []
			# for cluster in clusters:
			# 	cl = similarity_matrix[cluster][:, cluster]
			# 	print(len(cl))
			# 	maxx = np.max(cl)
			# 	c_diameters.append(maxx)
			max_cluster_dia = np.argmax(c_diameters)  #maximum cluster diameter
			max_difference_index = np.argmax(np.mean(similarity_matrix[clusters[max_cluster_dia]][:, clusters[max_cluster_dia]], axis=1))
			splinters = [clusters[max_cluster_dia][max_difference_index]] #spinter group
			last_clusters = clusters[max_cluster_dia]
			del last_clusters[max_difference_index]
			while True:
				split = False
				for j in range(len(last_clusters))[::-1]:
					splinter_distances = similarity_matrix[last_clusters[j], splinters]
					last_distances = similarity_matrix[last_clusters[j], np.delete(last_clusters, j, axis=0)]
					if np.mean(splinter_distances) <= np.mean(last_distances):
						splinters.append(last_clusters[j])
						del last_clusters[j]
						split = True
						break
				if split == False:
					break
			del clusters[max_cluster_dia]
			clusters.append(splinters)
			clusters.append(last_clusters)
			if len(clusters) == k:
				break

		cluster_labels = np.zeros(self.n_samples)
		for i in range(len(clusters)):
			cluster_labels[clusters[i]] = i

		return (clusters, cluster_labels)


	def purity(self,y_true,y_pred):
		np.seterr(divide='ignore', invalid='ignore')
		cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
		return np.sum(np.amax(cm,axis=0)) / np.sum(cm)

	def f_measure(self,y_true,y_pred):
		np.seterr(divide='ignore', invalid='ignore')
		cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
		tp = np.diag(cm)
		fp=cm.sum(axis=0) - np.diag(cm)
		fn = cm.sum(axis=1) - np.diag(cm)
		tn=cm.sum()-(tp+fn+fp)
		recall = tp/(tp+fn)
		precision = tp/(tp+fp)
		f1 = 2*((precision*recall)/(precision+recall))
		f1[np.isnan(f1)] = 0
		return f1.sum()


trainImagePath = "../dataset/train-images-idx3-ubyte"
trainLabelPath = "../dataset/train-labels-idx1-ubyte"
testImagePath = "../dataset/t10k-images-idx3-ubyte"
testLabelPath="../dataset/t10k-labels-idx1-ubyte"

x_train = idx2numpy.convert_from_file(trainImagePath)
x_test = idx2numpy.convert_from_file(testImagePath)

y_train = idx2numpy.convert_from_file(trainLabelPath)
y_test = idx2numpy.convert_from_file(testLabelPath)

x_train=x_train.reshape((x_train.shape[0], 28*28))
x_test=x_test.reshape((x_test.shape[0], 28*28))




Hx_train, Hy_train = ReduceDataSet.reduce(x_train,y_train,10,5000)


histArray=Histogram.histogram(Hx_train,16)

print ("-----------------------Hierarchical clustering algorithm---------------------------------")

print("########################R2-Data Represantation###########################")
Hmodel = Hierarchical(histArray)

print("-----------------Euclidean------------------")
Hclusters, Hlabels = Hmodel.fit(10, "Euclidean")
print("Purity score {:.3f}%".format(Hmodel.purity(y_true=Hy_train, y_pred=Hlabels)*100))
print("F-measure score {:.3f}%".format(Hmodel.f_measure(y_true=Hy_train, y_pred=Hlabels)*100))

print("-----------------Manhattan------------------")
Hclusters, Hlabels = Hmodel.fit(10, "Manhattan")
print("Purity score {:.3f}%".format(Hmodel.purity(y_true=Hy_train, y_pred=Hlabels)*100))
print("F-measure score {:.3f}%".format(Hmodel.f_measure(y_true=Hy_train, y_pred=Hlabels)*100))

print("-----------------Cosine------------------")
Hclusters, Hlabels = Hmodel.fit(10, "Cosine")
print("Purity score {:.3f}%".format(Hmodel.purity(y_true=Hy_train, y_pred=Hlabels)*100))
print("F-measure score {:.3f}%".format(Hmodel.f_measure(y_true=Hy_train, y_pred=Hlabels)*100))

print("########################R1-Data Represantation###########################")


x_train, y_train = ReduceDataSet.reduce(x_train,y_train,10,2000)
x_train=x_train/255.0

model = Hierarchical(x_train)

print("-----------------Euclidean------------------")
clusters, labels = model.fit(10, "Euclidean")
print("Purity score {:.3f}%".format(model.purity(y_true=y_train, y_pred=labels)*100))
print("F-measure score {:.3f}%".format(model.f_measure(y_true=y_train, y_pred=labels)*100))

print("-----------------Manhattan------------------")
clusters, labels = model.fit(10, "Manhattan")
print("Purity score {:.3f}%".format(model.purity(y_true=y_train, y_pred=labels)*100))
print("F-measure score {:.3f}%".format(model.f_measure(y_true=y_train, y_pred=labels)*100))

print("-----------------Cosine------------------")
clusters, labels = model.fit(10, "Cosine")
print("Purity score {:.3f}%".format(model.purity(y_true=y_train, y_pred=labels)*100))
print("F-measure score {:.3f}%".format(model.f_measure(y_true=y_train, y_pred=labels)*100))