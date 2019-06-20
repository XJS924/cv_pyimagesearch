#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simpledatasetloader.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-17 16:18   jiashuXu      1.0         None
'''
import cv2
import os
import numpy as np

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose> 0 and i>0 and (i+1)%verbose==0:
                print(f'[INFO] processed {i+1}/{len(imagePaths)}')

        return (np.array(data),np.array(labels))