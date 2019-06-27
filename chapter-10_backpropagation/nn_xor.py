#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nn_xor.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-20 20:07   jiashuXu      1.0         None
'''

from neuralnetwork import NeuralNetwork
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)
for x, target in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print('[INFO] data={},ground-truth={},pred={},step={}'.format(x, target[0], pred, step))
