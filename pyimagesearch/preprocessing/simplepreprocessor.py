#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simplepreprocessor.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-17 16:18   jiashuXu      1.0         None
'''
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image height/width. and interpolation
        # method used when resizing

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)