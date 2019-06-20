#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mini-batch-batch SGD.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-19 16:43   jiashuXu      1.0         None
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, w):
    preds = sigmoid_activation(X.dot(w))
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])


## 定义变量
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=float, default=100,
                help='# of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01,
                help='learning ratea')
ap.add_argument('-b', '--batch_size', type=int, default=32,
                help='size of SGD mini-batches')

args = vars(ap.parse_args())

## 生成样本点
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones((X.shape[0]))]

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=42)

print('[INFO] training......')

W = np.random.rand(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args['epochs']):
    epochLoss = []

    for batch_xs, batch_ys in next_batch(X, y, args['batch_size']):
        preds = sigmoid_activation(batch_xs.dot(W))

        error = preds - batch_ys

        epochLoss.append(np.sum(error ** 2))

        gradient = batch_xs.T.dot(error)

        W += -args['alpha'] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch % 0 == 0 or (epoch + 1) % 5 == 0:
        print('[INFO] epoch:{},loss={:.7f}'.format(epoch + 1, loss))

print('[INFO] evaluating....')
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', s=30)

plt.style.use('ggplot')

plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
