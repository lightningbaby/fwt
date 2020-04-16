#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   re_datamgr.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-04-16 13:43   tangyubao      1.0         None
'''

# import lib
 

# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from data.dataset import  SetDataset, MultiSetDataset, EpisodicBatchSampler, MultiEpisodicBatchSampler
from abc import abstractmethod
import torch.utils.data as data
from data.re_dataset import FewRelDataset
import json



# def collate_fn(data):
#     batch_data = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
#     batch_label = []
#     all_sets, all_labels = zip(*data)
#
#     for i in range(len(all_sets)):
#         for k in all_sets[i]:
#             batch_data[k] += all_sets[i][k]
#         batch_label += all_labels[i]
#     for k in batch_data:
#         batch_data[k] = torch.stack(batch_data[k], 0)
#     batch_label = torch.tensor(batch_label)
#     return batch_data,  batch_label
def collate_fn(data):
    batch_data = {'word': '', 'pos1': '', 'pos2': '', 'mask': ''}
    # batch_data = []
    batch_label = []
    all_sets, all_labels = zip(*data)

    for i in range(len(all_sets)):
        for k in all_sets[i]:
            batch_data[k] += all_sets[i][k]
        batch_label += all_labels[i]
    for k in range(len(batch_data)):
        batch_data[k] = torch.stack(batch_data[k], 0)
    batch_label = torch.tensor(batch_label)


    return batch_data,  batch_label


class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, max_length):
    pass

class SimpleREDataManager(DataManager):
  def __init__(self, batch_size):
    super(SimpleREDataManager, self).__init__()
    self.batch_size = batch_size
    try:
        # self.glove_mat = np.load('./pretrain/glove/glove_mat.npy')
        self.glove_word2id = json.load(open('./glove/glove_word2id.json'))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")

  def get_data_loader(self, data_file, max_length):
      dataset = FewRelDataset(data_file, self.glove_word2id, max_length)
      # data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True, collate_fn=collate_fn)
      data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)

      data_loader = data.DataLoader(dataset=dataset,
                                    **data_loader_params)
      return data_loader

  # def get_data_loader(self, data_file): #parameters that would change on train/val set
  #   dataset = SimpleDataset(data_file)
  #   data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
  #   data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
  #
  #   return data_loader





class SetDataManager(DataManager):
  def __init__(self, n_way, n_support, n_query, n_eposide=100):
    super(SetDataManager, self).__init__()
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_eposide = n_eposide


  def get_data_loader(self, data_file, max_length): #parameters that would change on train/val set
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_eposide )
    else:
      dataset = SetDataset( data_file , self.batch_size )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader
