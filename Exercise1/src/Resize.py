import numpy as np
def resize(dataset, width, length, nWidth, nLength):
    N = len(dataset)
    widthStep = int(width/nWidth)
    lengthStep = int(length/nLength)
    newDataset=list()
    print()
    for i in range(N):
        image = dataset[i]
        rImage=list()


        for j in range(length):
            row = list()
            for k in range(0,width, widthStep):
                pixel = 0.0
                for l in range(k,k+widthStep):
                    pixel+=image[j][l]
                row.append(pixel/nWidth)
            rImage.append(row)
        #print(rImage)
        ##Columns
        finalrImage=list()
        for j in range(nWidth):
            column = list()
            for k in range(0,length, lengthStep):
                pixel = 0
                for l in range(k, k+lengthStep):
                    pixel+=rImage[l][j]
                column.append(pixel/nLength)
            #column=np.array(column)
            finalrImage.append(column)
        finalrImage= np.array(finalrImage)
        newDataset.append(finalrImage.T)


    newDataset=np.array(newDataset)
    return newDataset



