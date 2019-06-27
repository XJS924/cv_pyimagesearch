#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-26 12:50   jiashuXu      1.0         None
'''
from concurrent import futures
import time


def test(num):
    # for i in range(5):
    #     print(i)
    time.sleep(1)
    return time.ctime(), num


data = list(range(100))
print(time.ctime())
with futures.ThreadPoolExecutor(max_workers=100) as executor:
    for future in executor.map(test, data):
        pass

print(time.ctime())