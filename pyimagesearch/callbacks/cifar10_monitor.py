#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cifar10_monitor.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-02 10:45   jiashuXu      1.0         None
'''

import matplotlib
matplotlib.use('Agg')

from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output directory')

args = vars(ap.parse_args())

print('[INFO] process ID:{}'.format(os.getpid()))

print('[INFO] loading CIFAR-10 data....')
(trainX,trainY),(testX,testY)= cifar10.load_data()

trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

lb= LabelBinarizer()

trainY,testY= lb.fit_transform(trainY),lb.fit_transform(testY)

labelNames = ['airplane','automobile','bird','cat','deer','dog','frog',
              'horse','ship','truck']

print('[INFO] compiling model...')
opt= SGD(lr= 0.01,momentum=0.9,nesterov=True)
model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

figPath = os.path.sep.join([args['output'],'{.png}'.format(os.getpid())])
jsonPath  = os.path.sep.join([args['output'],'{}.json'.format(os.getpid())])

callabcks = [TrainingMonitor(figPath,jsonPath=jsonPath)]

print('[INFO] training network...')

model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=100,
          callbacks=callabcks,verbose=1)



