#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   re_dataset.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-04-16 13:43   tangyubao      1.0         None
'''

# import lib
 
# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
import torch.utils.data as data

identity = lambda x:x


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, data_file, word2id, max_length):
        if not os.path.exists(data_file):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(data_file))
        self.classes = list(self.json_data[0])
        self.data = self.json_data[1]
        self.word2id = word2id
        self.max_length = max_length
        self.classes_dict={}

        for i in range(len(self.classes)):
            self.classes_dict[self.classes[i]]=i



    def __tokenize__(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.__tokenize__(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __getsent__(self,word,sent):
        for w in word:
            sent += w + ' '
        return sent

    def __getoridata__(self, item):
        word, pos1, pos2 = item['tokens'],item['h'][0],item['t'][0]

        sent = self.__getsent__(word,' ')
        p1 = '<'+pos1+ '>'
        p2 = '<'+pos2+ '>'

        return sent + p1 + p2

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        data=self.data[index]
        label=self.classes_dict[data['label']]
        word, pos1, pos2, mask = self.__getraw__(data)
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        # one_data=torch.stack([word,pos1,pos2,mask],0) # [4,128]
        one_data=torch.cat([word,pos1,pos2,mask],0) #[512]
        ori_data = self.__getoridata__(data)

        return one_data,label,ori_data

    def __len__(self):
        return len(self.data)

class SimpleDataset:
  def __init__(self, data_file):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    return img, target # img[3,224,224],target int

  def __len__(self):
    return len(self.meta['image_names'])


class SetDataset:
  def __init__(self, data_file, batch_size,max_length):
    self.max_length=max_length


    with open(data_file, 'r') as f:
      self.data = json.load(f)
    self.meta=self.data[1]

    # self.cl_list = np.unique(self.meta['image_labels']).tolist()
    self.classes=self.data[0]
    self.cl_list=[]
    for c in range(len(self.classes)):
        self.cl_list.append(c)

    self.classes_dict = {}

    for i in range(len(self.classes)):
        self.classes_dict[self.classes[i]] = i

    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []


    for i in range(len(self.meta)):
        self.sub_meta[self.classes_dict[self.meta[i]['label']]].append(self.meta[i])

    self.sub_dataloader = []
    sub_data_loader_params = dict(batch_size = batch_size,
        shuffle = True,
        num_workers = 0, #use main thread only or may receive multiple batches
        pin_memory = False)
    for cl in self.cl_list:
      sub_dataset = SubDataset(self.sub_meta[cl], cl,self.max_length)
      self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )




  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)


class MultiSetDataset:
  def __init__(self, data_files, batch_size, max_length):
    self.max_length=max_length
    self.cl_list = np.array([])
    self.sub_dataloader = []
    self.n_classes = []
    for data_file in data_files:
      with open(data_file, 'r') as f:
        data = json.load(f)
      meta=data[1]
      cl_list=data[0]
      # self.cl_list = np.concatenate((self.cl_list, cl_list))

      classes_dict = {}

      for i in range(len(cl_list)):
          classes_dict[cl_list[i]] = i

      sub_meta = {}
      for cl in cl_list:
        sub_meta[classes_dict[cl]] = []

      for i in range(len(meta)):
          sub_meta[classes_dict[meta[i]['label']]].append(meta[i])

      sub_data_loader_params = dict(batch_size = batch_size,
          shuffle = True,
          num_workers = 0, #use main thread only or may receive multiple batches
          pin_memory = False)
      for cl in cl_list:
        sub_dataset = SubDataset(sub_meta[classes_dict[cl]], cl, self.max_length)
        self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
      self.n_classes.append(len(cl_list))

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)

  def lens(self):
    return self.n_classes


class SubDataset:
  def __init__(self, sub_meta, cl,max_length):
    self.sub_meta = sub_meta
    self.cl = cl
    self.max_length=max_length
    try:
        # self.glove_mat = np.load('./pretrain/glove/glove_mat.npy')
        self.word2id = json.load(open('./glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")


  def __tokenize__(self, raw_tokens, pos_head, pos_tail):
    # token -> index
    indexed_tokens = []
    for token in raw_tokens:
        token = token.lower()
        if token in self.word2id:
            indexed_tokens.append(self.word2id[token])
        else:
            indexed_tokens.append(self.word2id['[UNK]'])

    # padding
    while len(indexed_tokens) < self.max_length:
        indexed_tokens.append(self.word2id['[PAD]'])
    indexed_tokens = indexed_tokens[:self.max_length]

    # pos
    pos1 = np.zeros((self.max_length), dtype=np.int32)
    pos2 = np.zeros((self.max_length), dtype=np.int32)
    pos1_in_index = min(self.max_length, pos_head[0])
    pos2_in_index = min(self.max_length, pos_tail[0])
    for i in range(self.max_length):
        pos1[i] = i - pos1_in_index + self.max_length
        pos2[i] = i - pos2_in_index + self.max_length

    # mask
    mask = np.zeros((self.max_length), dtype=np.int32)
    mask[:len(indexed_tokens)] = 1

    return indexed_tokens, pos1, pos2, mask

  def __getraw__(self, item):
    word, pos1, pos2, mask = self.__tokenize__(item['tokens'],
                                                   item['h'][2][0],
                                                   item['t'][2][0])
    return word, pos1, pos2, mask

  def __getsent__(self, word, sent):
      for w in word:
          sent += w + ' '
      return sent

  def __getoridata__(self, item):
      word, pos1, pos2 = item['tokens'], item['h'][0], item['t'][0]

      sent = self.__getsent__(word, ' ')
      p1 = '<' + pos1 + '>'
      p2 = '<' + pos2 + '>'

      return sent + p1 + p2

  def __getitem__(self,i):
    label=self.cl
    data=self.sub_meta[i]
    word, pos1, pos2, mask = self.__getraw__(data)
    word = torch.tensor(word).long()
    pos1 = torch.tensor(pos1).long()
    pos2 = torch.tensor(pos2).long()
    mask = torch.tensor(mask).long()
    one_data = torch.cat([word, pos1, pos2, mask], 0)  # [512]
    ori_data = self.__getoridata__(data)
    return one_data, label,ori_data

  def __len__(self):
    return len(self.sub_meta)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_classes)[:self.n_way]


class MultiEpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes
    self.n_domains = len(n_classes)

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    domain_list = [i%self.n_domains for i in range(self.n_episodes)]
    random.shuffle(domain_list)
    for i in range(self.n_episodes):
      domain_idx = domain_list[i]
      start_idx = sum(self.n_classes[:domain_idx])
      yield torch.randperm(self.n_classes[domain_idx])[:self.n_way] + start_idx

