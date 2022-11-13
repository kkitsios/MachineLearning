import numpy as np
def reduce(X,Y,k, n):
    dataset = []
    labels =[]
    for i in  range(k):
        c=n//k
        j=0
        while c>0:
            if Y[j] == i:
                dataset.append(X[j])
                labels.append(Y[j])
                c-=1
            j+=1
    return np.array(dataset), np.array(labels)