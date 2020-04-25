#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   split_dataset.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-04-25 22:35   tangyubao      1.0         None
'''

# import lib
 
import json
import random
path='../re_data/val_pubmed.json'
out1='../re_data/sub_val_pubmed.json'
out2='../re_data/sub_test_pubmed.json'
data=json.load(open(path,'r'))
classes=data.keys()
test_class=random.sample(classes,4)
sub1={}
sub2={}
for c in classes:
  if c in test_class:
    sub2[c]=data[c]
  else:
    sub1[c]=data[c]
#print(len(sub1))
#print(sub2.keys())
#print(test_class)

json.dump(sub1,open(out1,'w'))
json.dump(sub2,open(out2,'w'))

