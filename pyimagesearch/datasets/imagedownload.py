#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   imagedownload.py    
@Contact :   jiashu42@sina.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-06-26 09:30   jiashuXu      1.0         None
'''

import requests
from PIL import Image
import io
import time
import random
from urllib import request
import codecs
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

data = []
animals = ['cat.txt','dog.txt','panda.txt']
for txt in animals:
    t  =()
    animal = txt.split('.')[0]
    with codecs.open(txt,'r') as fr:
        for idx,url in enumerate(fr.read().split('\n')):
            data.append((url,animal,idx))

# print(data)
def downloading():
    for param in data:
        try:
            imgUrl,animal,idx= param
            path = './animals/{}{:03}.jpg'.format(animal, idx)
            # request.urlretrieve(imgUrl,path)
            img = requests.get(imgUrl,timeout=10,headers= {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}).content
            time.sleep(random.randint(1,4))
            img = Image.open(io.BytesIO(img))
            img.save(path)
        except Exception as e:
            print(imgUrl,e)
            pass

# with ThreadPoolExecutor(max_workers=5) as executor:
#     for future in executor.map(downloading,data):
#         print(future)

# with requests.session() as sess:
# header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
#           }
# sess.headers.update(header)

if __name__=='__main__':
    downloading()



# for animal, url in urlDict.items():
#     content = requests.get(url).text
#     imageUrlList = content.split('\r\n')[:1000]
#     print(animal,imageUrlList[:2])
#     for idx, imgUrl in enumerate(imageUrlList[:]):
#         try:
#             path = './animals/{}{:03}.jpg'.format(animal, idx)
#             img = request.urlretrieve(imgUrl,path)
#             # img = Image.open(io.BytesIO(img))
#             if (idx + 1) % 100 == 0:
#                 print('[INFO] downloading {} images'.format((idx + 1) * 100))
#             # img.save()
#         except Exception as e:
#             print(imgUrl)
