#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visualize_architecture.py.py
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-02 15:23   jiashuXu      1.0         None
'''

from pyimagesearch.nn.conv.lenet import LeNet
from keras.utils import plot_model

model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file='lenet.png', show_shapes=True)
