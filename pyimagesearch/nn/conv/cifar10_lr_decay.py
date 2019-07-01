#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cifar10_lr_decay.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-30 21:15   jiashuXu      1.0         None
'''

import matplotlib

matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    return float(alpha)


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='path to the output loss/accuracy plt')
args = vars(ap.parse_args())

print('[INOF] loading CIFAR-10 data....')
trainX, trainY, testX, testY = cifar10.load_data()
testX, trainX = testX.asplot('float') / 255.0, trainX.astype('float') / 255.0

lb = LabelBinarizer()
trainY, testY = lb.fit_transform(trainY), lb.fit_transform(testY)

labelNames = ['airplane','automobile','bird','deer','cat','dog','frog','horse','ship','truck']

callbacks  =[LearningRateScheduler(step_decay)]

opt= SGD(lr= 0.01,momentum=0.9,nesterov=True)

model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt)

H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,
              epohcs=40,callbacks=callbacks)

print('[INFO] evaluating network....')

predictions = model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,40),H.history['loss'],label='training_loss')
plt.plot(np.arange(0,40),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,40),H.history['acc'],label='training_acc')
plt.plot(np.arange(0,40),H.history['val_acc'],label='val_acc')

plt.title('Training Loss and Accuracy on CIFAR-10 data ')
plt.xlabel('Epoch # ')
plt.ylabel('Loss /Accuracy')

plt.legend()
plt.savefig(args['output'])



