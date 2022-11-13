import idx2numpy
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import Resize

trainImagePath = "../dataset/train-images-idx3-ubyte"
trainLabelPath = "../dataset/train-labels-idx1-ubyte"
testImagePath = "../dataset/t10k-images-idx3-ubyte"
testLabelPath="../dataset/t10k-labels-idx1-ubyte"

x_train = idx2numpy.convert_from_file(trainImagePath)
y_train = idx2numpy.convert_from_file(trainLabelPath)

x_test = idx2numpy.convert_from_file(testImagePath)
y_test = idx2numpy.convert_from_file(testLabelPath)


x_train = x_train / 255.0
x_test = x_test / 255.0

x_test=x_test.reshape((x_test.shape[0], 28*28))

model = tf.keras.Sequential([
   tf.keras.layers.Flatten(input_shape=(28,28)),
   tf.keras.layers.Dense(500, activation='sigmoid'),
   tf.keras.layers.Dense(10)
])

model.compile(optimizer='sgd',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, epochs=10)


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
prediction = probability_model.predict(x_test)

predictions=[]
for i in range(prediction.shape[0]):
   predictions.append(np.argmax(prediction[i]))
print("One hidden layer")
print("acc={:.3f}%".format(metrics.accuracy_score(y_test, predictions)*100.0))
print("f1 score={:.3f}%".format(metrics.f1_score(y_true=y_test, y_pred=predictions, average='macro')*100.0))


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(500, activation='sigmoid'),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='sgd',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, epochs=10)


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
prediction = probability_model.predict(x_test)

predictions=[]
for i in range(prediction.shape[0]):
    predictions.append(np.argmax(prediction[i]))
print("Two hidden layers")
print("acc={:.3f}%".format(metrics.accuracy_score(y_test, predictions)*100.0))
print("f1 score={:.3f}%".format(metrics.f1_score(y_true=y_test, y_pred=predictions, average='macro')*100.0))
#acc=79.820
#f1=79.762

#acc=82.420
#f1=82.309

