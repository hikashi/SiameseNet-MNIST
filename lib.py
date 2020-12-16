# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:48:14 2020

@author: hikashi
"""
import idx2numpy
import numpy as np
# import keras
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import seaborn as sns

def loadData(Directory):
    x_train  = idx2numpy.convert_from_file(Directory + 'train-images.idx3-ubyte')
    y_train  = idx2numpy.convert_from_file(Directory + 'train-labels.idx1-ubyte')
    x_test   = idx2numpy.convert_from_file(Directory + 't10k-images.idx3-ubyte')
    y_test   = idx2numpy.convert_from_file(Directory + 't10k-labels.idx1-ubyte')
    return x_train, y_train, x_test, y_test

def prepData(x_train, y_train, x_test, y_test):
    # prep Data
    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_test  = x_test.reshape(len(x_test), 28, 28, 1)

    # perform normalization on the dataset
    x_train = x_train / 255
    x_test  = x_test / 255

    return x_train, y_train, x_test, y_test

def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["binary_accuracy"], label="train_acc")
    plt.plot(H.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath + 'trainingOut.pdf')
