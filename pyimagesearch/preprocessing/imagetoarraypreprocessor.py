#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   imagetoarraypreprocessor.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-25 18:00   jiashuXu      1.0         None
'''

from keras.preprocessing.image import img_to_array
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader

class ImageToArrayPreprocessor:
    def __init__(self,dataFormat = None):
        self.dataFormat = dataFormat

    def preprocess(self,image):
        return img_to_array(image,data_format=self.dataFormat)

if __name__=='__main__':
    sp = SimplePreprocessor(32,32)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
    data,labels = sdl.load(imagePaths,verbose=500)

