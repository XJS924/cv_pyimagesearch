#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   knn_image_classifier.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-17 18:27   jiashuXu      1.0         None
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-K', '--neighbors', type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=1,
                help='# of jobs for K-NN distance (-1 uses alla available cores)')
args = vars(ap.parse_args())

print('[INFO] loading images....')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

print('[INFO] featuress matrix:{:.1f}MB'.format(data.nbytes / (1024 * 1000)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

print('[INFO] evaluating k-NN classifier...')

model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))



