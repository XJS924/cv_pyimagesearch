#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   shallownet_animals.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-25 19:54   jiashuXu      1.0         None
'''

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.shallownet import ShallowNet

from keras.optimizers import SGD
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
args = vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelEncoder().fit_transform(trainY)
testY = LabelEncoder().fit_transform(testY)

print('[INFO] compiling model...')
opt = SGD(0.005)
model = ShallowNet.build(32, 32, 3, 3)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbpse=100)

print('[INFO] evaluating networks....')
predictions=  model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),
                            target_names=['cat','dog','pandas']))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100),H.history['loss'],label='training_loss ')
plt.plot(np.arange(0,100),H.history['val_loss'],label='val_loss ')
plt.plot(np.arange(0,100),H.history['acc'],label='training_acc ')
plt.plot(np.arange(0,100),H.history['val_acc'],label='val_acc ')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()




