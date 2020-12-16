# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:50:35 2020

@author: hikashi
"""
import cv2 as cv
import numpy as np
from tensorflow.keras import models

inputTestData1  = 'TestData/test7.png'
image1 = cv.imread(inputTestData1, cv.IMREAD_GRAYSCALE)
image1 = cv.resize(image1, (28,28))
image1 = 255-image1          #inverts image. Always gets read inverted.
image1 = image1 / 255

inputTestData2  = 'TestData/test8.png'
image2 = cv.imread(inputTestData2, cv.IMREAD_GRAYSCALE)
image2 = cv.resize(image2, (28,28))
image2 = 255-image2          #inverts image. Always gets read inverted.
image2 = image2 / 255

inputTestData3  = 'TestData/test1.png'
image3 = cv.imread(inputTestData3, cv.IMREAD_GRAYSCALE)
image3 = cv.resize(image3, (28,28))
image3 = 255-image3          #inverts image. Always gets read inverted.
image3 = image2 / 255

SiamModel = models.load_model('model/siamModel.h5')

result = SiamModel.predict([image1.reshape(1, 28, 28, 1), image2.reshape(1, 28, 28, 1)])
print(result[0][0])

result = SiamModel.predict([image1.reshape(1, 28, 28, 1), image3.reshape(1, 28, 28, 1)])
print(result[0][0])

result = SiamModel.predict([image2.reshape(1, 28, 28, 1), image3.reshape(1, 28, 28, 1)])
print(result[0][0])