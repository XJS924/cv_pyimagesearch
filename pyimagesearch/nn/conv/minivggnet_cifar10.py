#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   minivggnet_cifar10.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-27 20:18   jiashuXu      1.0         None
'''

import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap  = argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to output loss / accuracy plot')
args = vars(ap.parse_args())

print('[INFO] loading CIFAR-10 data....')
trainX,trainY,testX,testY = cifar10.load_data()

trainX,testX = trainX.astype('float')/255.0,testX.astype('float')/255.0
lb = LabelBinarizer()
trainY,testY = lb.fit_transform(trainY),lb.fit_transform(testY)

lableNames= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship'
             'truck']

print('[INFO] compiling model....')
opt = SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov=True)

model =MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt)

print('[INFO] training network...')
H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,
              epochs=40,verbose=1)

print('[INFO] evaluating network...')
predictions = model.predict(testX,batch_size=64)

print(classification_report(testY.argmax(axis=1),predictions.argmax(aixs=1),target_names=lableNames))


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,40),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,40),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,40),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,40),H.history['val_acc'],label='val_acc')

plt.title('Training Loss and Accuracy on CIFAR-10')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['output'])



