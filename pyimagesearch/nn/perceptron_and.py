#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   perceptron_and.py.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-20 14:54   jiashuXu      1.0         None
'''

from pyimagesearch.nn.perceptron import Perceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
# y =np.array([[0],[0],[0],[1]]) and
y = np.array([[0],[1],[1],[0]]) # XOR
print('[INFO] perceptron train....')
p = Perceptron(X.shape[1],alpha=0.1)
p.fit(X,y,epochs=20)

print('[INFO] perceptron testing....')
for x,target in zip(X,y):
    pred = p.predict(x)
    print('[INFO] data={},ground-truth={},pred={}'.format(x,target[0],pred))