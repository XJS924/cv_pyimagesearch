#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cifar10_checkpoint_best.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-02 15:03   jiashuXu      1.0         None
'''

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-w','--weights',required=True,
                help='path to best model weights file')
args = vars(ap.parse_args())

print('[INFO] loading CIFAR-10 data。。。')
(trainX,trainY),(testX,testY) = cifar10.load_data()

lb = LabelBinarizer()
trainX,testX = trainX.astype('float')/255.0,testX.astype('float')/255.0
trainY,testY = lb.fit_transform(trainY),lb.fit_transform(testY)

print('[INFO] compiling model...')
opt= SGD(lr=0.01,decay =0.01/40,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
checkpoint = ModelCheckpoint(args['weights'],monitor='val_loss',
                             save_best_only=True,verbose=1)
callbacks=[checkpoint]

print('[INFO] training network...')
H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,
              epochs=40,callbacks=callbacks,verbose=1)

