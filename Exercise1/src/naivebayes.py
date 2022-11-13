from math import sqrt, exp, pi, pow
import idx2numpy
from sklearn import metrics
from Resize import resize
#Calculate mean.
def mean(numbers):
	return sum(numbers)/float(len(numbers)) if sum(numbers)/float(len(numbers)) !=0 else pow(10,-20)

 #Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return sqrt(variance)

# Split the dataset by class values, returns a dictionary
def separate_by_class(X,Class):
	separated = dict()
	for i in range(len(X)):
		vector = X[i]
		class_value = Class[i]

		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def fit(X,Class):
    summaries = dict()
    seperated = separate_by_class(X,Class);
    for class_value, rows in seperated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
 
# Predict the class for a given row
def predict(summaries, row):	
	predictions=list()
	for i in range(len(row)):
		probabilities = calculate_class_probabilities(summaries, row[i])
		best_label, best_prob = None, -1
		for class_value, probability in probabilities.items():
			if best_label is None or probability > best_prob:
				best_prob = probability
				best_label = class_value
		predictions.append(best_label)		
	return predictions    


#Main
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



x_train=x_train.reshape((x_train.shape[0], 7 * 7))
x_test=x_test.reshape((x_test.shape[0], 7*7))

x_train=x_train/255.0
x_test=x_test/255.0

model = fit(x_train,y_train)

predictions=predict(model,x_test)
print("acc={:.3f}%".format(metrics.accuracy_score(y_test, predictions)*100.0))
print("f1 score={:.3f}%".format(metrics.f1_score(y_true=y_test, y_pred=predictions, average='macro')*100.0))

#accuracy=60.07
#f1_score=57.285

