#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainingmonitor.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-01 18:12   jiashuXu      1.0         None
'''

from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs=None):
        self.H = {}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath))

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.jsonPath is not None:
            f = open(self.jsonPath,'w')
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H['loss'])>1:
            N = np.arange(0,len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N,self.H['loss'],label='training_loss')
            plt.plot(N,self.H['val_loss'],label='val_loss')
            plt.plot(N,self.H['acc'],label='acc')
            plt.plot(N,self.H['val_acc'],label='val_acc')

            plt.title('Training Loss and Accuracy [Epoch {}]'.format(len(self.H['loss'])))

            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')

            plt.savefig(self.figPath)
            plt.close()




