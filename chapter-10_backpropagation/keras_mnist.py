#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   keras_mnist.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-22 09:56   jiashuXu      1.0         None
'''

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,Softmax
from keras.optimizers import  SGD
from sklearn import datasets
import  matplotlib.pyplot  as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o','--output',required=False,
                help = 'path to the output loss/accuracy plot')

args = vars(ap.parse_args())

print('[INFO] loading mnist full dataset ...')

dataset = datasets.fetch_mldata('MNIST Original')
data = dataset.data.astype('float')/255.0

trainX,testX,trainY,testY = train_test_split(data,dataset.target,test_size=0.25)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

#建模

model = Sequential()
model.add(Dense(256,input_shape=(784,),activation='sigmoid'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))

print('[INFO] training networking...')

sgd = SGD(0.01)
model.compile(loss= 'categorical_corssentropy',optimizer=sgd,accuracy=['categorical_accuracy'])

print(model.summary())
model.fit(trainX,trainY,validation_data=[testX,testY],epochs= 100,
              batch_size=128)



