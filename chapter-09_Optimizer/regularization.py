#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   regularization.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-20 09:29   jiashuXu      1.0         None
'''

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d",'--dataset',required=True,
                help = 'path to input dataset')
args = vars(ap.parse_args())

print('[INFO] loading images....')

imagePaths = list(paths.list_images(args['dataset' ]))

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
data,labels = sdl.load(imagePaths,verbose=500)
data= data.reshape((data.shape[0],3072))

le= LabelEncoder()
labels = le.fit_transform(labels)
trainX,testX,trainY,testY = train_test_split(data,labels,test_size=0.25,random_state=42)

for r in (None,'l1','l2'):
    print("[INFO] train model with {} penalty".format(r))
    model = SGDClassifier(loss='log',penalty=r,max_iter=10,learning_rate='constant',
                          eta0=0.01,random_state=42)
    model.fit(trainX,trainY)
    acc = model.score(testX,testY)
    print('[INFO] {} penalty accuracy:{:.2f}%'.format(r,acc*100))

