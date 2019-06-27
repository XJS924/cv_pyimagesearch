#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   shallownet.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-25 19:48   jiashuXu      1.0         None
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width,height,depth,classes):

        model = Sequential()
        inputShape = (height,width,depth)

        if K.image_data_format()=='channels_first':
            inputShape=(depth,height,width)
        model.add(Conv2D(32,(3,3),padding='SAME',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

