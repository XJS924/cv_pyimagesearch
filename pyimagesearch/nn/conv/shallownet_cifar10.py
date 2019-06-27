#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   shallownet_cifar10.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-27 16:33   jiashuXu      1.0         None
'''

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

print('[INFO] laoading CIFAR-10 data...')
trainX, trainY, testX, testY = cifar10.load_data()
trainX, testX = trainX.astype('float') / 255.0, testX.astype('float') / 255.0

lb = LabelBinarizer()
trainY, testY = lb.fit_transform(trainY), lb.fit_transform(testY)

labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('[INFO] compile model...')
opt = SGD(0.01)
model = ShallowNet.build(32, 32, 3, 10)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

print('[INFO] training network....')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=40, verbose=1)

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label='training_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 40), H.history['acc'], label='acc')
plt.plot(np.arange(0, 40), H.history['val_acc'], label='val_acc')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()


