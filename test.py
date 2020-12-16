# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:46:08 2020

@author: hikashi
"""
import numpy as np
import lib
import loadMNISTData
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def init_cnn(inputShape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1120, activation='sigmoid'))
    return model

def siamesenet(inputShape):
    left_input  = layers.Input(inputShape)
    right_input = layers.Input(inputShape)

    cnn_model = init_cnn(inputShape)
    encodedImage1 = cnn_model(left_input)
    encodedImage2 = cnn_model(right_input)

    # calculate the distance
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encodedImage1, encodedImage2])

    # Same class or not prediction
    prediction = layers.Dense(units=1, activation='sigmoid')(l1_distance)
    model = models.Model(inputs=[left_input, right_input], outputs=prediction)

    # Define the optimizer and compile the model
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer="adam")
    return model

############################################################
# load the data from the Data directory
# x_train, y_train, x_test, y_test = lib.loadData('Data/')
trainData, testData = loadMNISTData.load_data()
x_train = trainData[0]
y_train = trainData[1]
x_test  = testData[0]
y_test  = testData[1]

# prep data
x_train, y_train, x_test, y_test = lib.prepData(x_train, y_train, x_test, y_test)

# add a channel dimension to the images
trainX = np.expand_dims(x_train, axis=-1)
testX = np.expand_dims(x_test, axis=-1)

# make pairwise data
(pairTrain, labelTrain) = lib.make_pairs(trainX, y_train)
(pairTest, labelTest)   = lib.make_pairs(testX, y_test)

# build the model
siamModel = siamesenet((28,28,1))

# attempt to fit the data
batch_size   = 32
epochs       = 10
histories = siamModel.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
                    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
                    batch_size=batch_size,epochs=epochs)

# output the score
score = siamModel.evaluate([pairTest[:, 0], pairTest[:, 1]], labelTest[:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


lib.plot_training(histories, 'result/')

siamModel.save('model/siamModel.h5')
