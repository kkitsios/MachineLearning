import numpy as np
def histogram(X,bins):
	histArray=[]
	for i in X:
		histArray.append(np.array(np.histogram(i,bins)[0]))
	return np.array(histArray, dtype='float64')

