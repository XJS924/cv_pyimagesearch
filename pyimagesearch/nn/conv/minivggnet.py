#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   minivggnet.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-27 20:03   jiashuXu      1.0         None
'''

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten,Dropout,Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format()=='channels_first':
            inputShape=(depth,height,width)
            chanDim=1

        model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model



