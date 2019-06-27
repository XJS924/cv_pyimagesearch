#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   keras_cifar10.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-22 10:31   jiashuXu      1.0         None
'''

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

print('[INFO] loading CIFAR-10 data...')
trainX, trainY, testX, testY = cifar10.load_data()
trainX, testX = trainX.astype('float') / 255.0, testX.astype('float') / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.reshape[0], 3072))

lb = LabelBinarizer()
trainY, testY = lb.fit_transform(trainY), lb.fit_transform(testY)
lableNames = ['airplane', 'automobile', 'bird', 'cat', 'eer', 'dog', 'frog', 'horse', 'ship',
              'truck']
