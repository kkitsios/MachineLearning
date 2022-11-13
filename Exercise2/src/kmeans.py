import idx2numpy as idx2numpy
import numpy as np
import Histogram
from sklearn.metrics import confusion_matrix


trainImagePath = "../dataset/train-images-idx3-ubyte"
trainLabelPath = "../dataset/train-labels-idx1-ubyte"

testImagePath = "../dataset/t10k-images-idx3-ubyte"
testLabelPath="../dataset/t10k-labels-idx1-ubyte"

x_train = idx2numpy.convert_from_file(trainImagePath)
y_train = idx2numpy.convert_from_file(trainLabelPath)

x_test = idx2numpy.convert_from_file(testImagePath)
y_test = idx2numpy.convert_from_file(testLabelPath)

x_train=x_train.reshape((x_train.shape[0], 28*28))
x_test=x_test.reshape((x_test.shape[0], 28*28))

x_train=x_train/255.0
x_test=x_test/255.0

k=10; # number of clusters
centroids = {}
def fit(data, distance):

    for i in range(k):
        centroids[i] = data[i]

    while(True):
        classifications = {}

        for i in range(k):
            classifications[i] = []

        for featureset in data:
            #TODO
            #Disntances!!
            if (distance=='Euclidean'):
                distances = [np.linalg.norm(featureset-centroids[centroid]) for centroid in  centroids]
            elif (distance == 'Manhattan'):
                distances = [np.linalg.norm(featureset - centroids[centroid], ord=1) for centroid in centroids]
            elif (distance == 'Cosine'):
                distances = []
                #distances = [np.dot(featureset.T, centroids[centroid])/(np.linalg.norm(featureset)*np.linalg.norm(centroids[centroid])) for centroid in centroids]
                for centroid in centroids:
                    dotP = np.dot(featureset, centroids[centroid].T)
                    #print(type(centroids[centroid]))
                    if (isinstance(centroids[centroid], np.float64)):dotP = 0
                    norm1 = np.linalg.norm(featureset)
                    if (np.isnan(norm1).any()): norm1 = 10**10
                    norm2 = np.linalg.norm(centroids[centroid])
                    if (np.isnan(norm2).any()):  norm2 = 10**10
                    distances.append(dotP/(norm2*norm1))


            #print(distances)
            minn=np.amin(distances)
            #print(minn)
            #exit(0)
            classification = distances.index(minn)
            classifications[classification].append(featureset)

        prev_centroids = dict(centroids)

        for classification in  classifications:
            centroids[classification] = np.average(classifications[classification],axis=0)

        optimized = True

        for c in  centroids:
            original_centroid = prev_centroids[c]
            current_centroid =  centroids[c]
            if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > 0.001:
                #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                optimized = False

        if optimized:
            break

def predict(data):
    classification = []
    for d in data:
        distances = [np.linalg.norm(d-centroids[centroid]) for centroid in centroids]
        classification.append(distances.index(min(distances)))
    return classification

def purity(y_true,y_pred):
        np.seterr(divide='ignore', invalid='ignore')
        cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
        return np.sum(np.amax(cm,axis=0)) / np.sum(cm)

def f_measure(y_true,y_pred):
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
print ("-----------------------Kmeans clustering algorithm---------------------------------")
print("########################R1-Data Represantation###########################")
fit(x_train, "Euclidean")
classes = predict(y_test)
print("-----------------Euclidean------------------")
print("Purity score {:.3f}%".format(purity(y_test,np.array(classes))*100))
print("F-measure score {:.3f}%".format(f_measure(y_true=y_test, y_pred=np.array(classes))*100))
fit(x_train, "Manhattan")
classes = predict(y_test)
print("-----------------Manhattan------------------")
print("Purity score {:.3f}%".format(purity(y_test,np.array(classes))*100))
print("F-measure score {:.3f}%".format(f_measure(y_true=y_test, y_pred=np.array(classes))*100))
fit(x_train, "Cosine")
classes = predict(y_test)
print("-----------------Cosine------------------")
print("Purity score {:.3f}%".format(purity(y_test,np.array(classes))*100))
print("F-measure score {:.3f}%".format(f_measure(y_true=y_test, y_pred=np.array(classes))*100))

print("########################R2-Data Represantation###########################")

histArray=Histogram.histogram(x_train,32)
testhistArray=Histogram.histogram(x_test,32)

fit(histArray, "Euclidean")
clusters=predict(testhistArray)
print("-----------------Euclidean------------------")
print("Purity score {:.3f}%".format(purity(y_test,np.array(clusters))*100))
print("F-measure score {:.3f}%".format(f_measure(y_true=y_test, y_pred=np.array(clusters))*100))

fit(histArray, "Manhattan")
clusters=predict(testhistArray)
print("-----------------Manhattan------------------")
print("Purity score {:.3f}%".format(purity(y_test,np.array(clusters))*100))
print("F-measure score {:.3f}%".format(f_measure(y_true=y_test, y_pred=np.array(clusters))*100))

fit(histArray, "Cosine")
clusters=predict(testhistArray)
print("-----------------Cosine------------------")
print("Purity score {:.3f}%".format(purity(y_test,np.array(clusters))*100))
print("F-measure score {:.3f}%".format(f_measure(y_true=y_test, y_pred=np.array(clusters))*100))