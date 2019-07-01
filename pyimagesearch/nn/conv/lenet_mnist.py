#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lenet_mnist.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-27 19:19   jiashuXu      1.0         None
'''

from pyimagesearch.nn.conv.lenet import LeNet
from keras.optimizers import  SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print('[INFO] access MNIST...')
dataset = datasets.fetch_mldata('MNIST Original')
data = dataset.data

if K.image_data_format()=='channels_first':
    data = data.reshape(data.shape[0],1,28,28)
else:
    data = data.reshape(data.shape[0],28,28,1)

trainX,testX,trainY,testY = train_test_split(data/255.0,dataset.target.astype('int'),
                                             test_size = 0.25,random_state=42)
le = LabelBinarizer()
trainY,testY = le.fit_transform(trainY),le.fit_transform(testY)

print('[INFO] compiling model...')
opt= SGD(0.01)
model = LeNet.build(width=28,height=28,depth=1,classes=10)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt)

print('[INFO] training network....')
H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=128,
              epochs=20,verbose=1)

print('[INFO] evaluating network....')
predictions = model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 20), H.history['loss'], label='training_loss')
plt.plot(np.arange(0, 20), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 20), H.history['acc'], label='acc')
plt.plot(np.arange(0, 20), H.history['val_acc'], label='val_acc')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

