#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lenet.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-27 18:56   jiashuXu      1.0         None
'''
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Dense,Flatten,Input,Activation
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        inputShape = (height,width,depth)
        if K.image_data_format()=='channels_first':
            inputShape = (depth,height,width)

        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size= (2,2),strides=(2,2)))

        model.add(Conv2D(50,(5,5),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model




