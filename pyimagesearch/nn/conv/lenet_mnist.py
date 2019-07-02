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
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print('[INFO] access MNIST...')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows = img_cols = 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
le = LabelBinarizer()
trainY,testY = le.fit_transform(y_train),le.fit_transform(y_test)

print('[INFO] compiling model...')
opt= SGD(0.01)
model = LeNet.build(width=28,height=28,depth=1,classes=10)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt)

print('[INFO] training network....')
H = model.fit(x_train,trainY,validation_data=(x_test,testY),batch_size=128,
              epochs=20,verbose=1)

print('[INFO] evaluating network....')
predictions = model.predict(x_test,batch_size=128)
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

