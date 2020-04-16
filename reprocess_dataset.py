#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   t.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-03-31 13:43   tangyubao      1.0         None
'''

# import lib

import json
in_path=''
out_path=''

data=json.load(open(in_path))
all_data=[]
all_all_data=[]
rel=list(data.keys())
rel_dict={}
for i in range(len(rel)):
  rel_dict[rel[i]]=i

for r in rel:
  # label=rel_dict[r]
  for d in data[r]:
    each_dict = {'label': [], 'h': [], 'tokens': [], 't': []}
    each_dict['label'] = r
    each_dict['h'] = d['h']
    each_dict['tokens'] = d['tokens']
    each_dict['t']=d['t']
    all_data.append(each_dict)
print(len(all_data))
all_all_data.append(rel)
all_all_data.append(all_data)
with open(out_path, 'w') as f:
  json.dump(all_all_data, f)
