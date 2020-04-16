#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_encoder.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-04-15 22:34   tangyubao      1.0         None
'''

# import lib
import numpy as np
import torch
import torch.optim
import os
import json
from methods import backbone
from methods.re_backbone import model_dict
from data.re_datamgr import  SimpleREDataManager
from data.datamgr import SetDataManager
from methods.re_baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.gnnnet import GnnNet
from re_args import parse_args, get_resume_file, load_warmup_state
from toolkit import data_preprocess, sentence_encoder, encoder

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):

  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(model.parameters())
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc = 0
  total_it = 0

  # start
  for epoch in range(start_epoch,stop_epoch):
    model.train() # resnet 完成了训练 #训练分类模型，先resnet提取特征、FCE后NLLloss分类
    total_it = model.train_loop(epoch, base_loader,  optimizer, total_it) #model are called by reference, no need to return
    model.eval()

    acc = model.test_loop( val_loader)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model
# --- main function ---
if __name__=='__main__':



  # set numpy random seed
  np.random.seed(10)

  # parser argument
  params = parse_args('train')  #model_dict=conv4, conv6, resnet10, resnet18,resnet34
  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)
  batch_size= params.batch_size


  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')


  if params.dataset == 'multi':
    print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
    # datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
    datasets = ['train_wiki', 'nyt', 'semeval','pubmed']
    datasets.remove(params.testset)
    base_file = [os.path.join(params.data_dir, params.train, '.json') for dataset in datasets]
    val_file  = os.path.join(params.data_dir, params.val,'.json')
  else:
    print('  train with single seen domain {}'.format(params.dataset))
    base_file  = os.path.join(params.data_dir,  (params.dataset + '.json'))
    val_file   = os.path.join(params.data_dir, (params.val + '.json'))

  # model
  print('\n--- build model ---')

  if params.method in ['baseline', 'baseline++'] : # paras.model resnet10 这是feature encoder
    print('  pre-training the feature encoder {} using method {}'.format(params.model, params.method))
    base_datamgr = SimpleREDataManager(batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, params.max_length)
    val_datamgr = SimpleREDataManager(batch_size)
    val_loader = val_datamgr.get_data_loader(val_file, params.max_length)

    if params.method == 'baseline':
      model           = BaselineTrain(model_dict[params.model], params.num_classes, tf_path=params.tf_dir)
    elif params.method == 'baseline++':

      model           = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax', tf_path=params.tf_dir)

  elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'gnnnet']:# 这是分类模型
    print('  baseline training the model {} with feature encoder {}'.format(params.method, params.model))

    #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    # n_query = max(1, int(16* params.test_n_way/params.train_n_way)) # 16

    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, batch_size=batch_size, n_query=params.n_query)  # （5，5）
    base_datamgr            = SetDataManager(**train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader(base_file)

    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, batch_size=batch_size, n_query=params.n_query)  # （5，5）
    val_datamgr             = SetDataManager(**test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader(val_file)

    if params.method == 'protonet':
      model           = ProtoNet( model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)
    elif params.method == 'gnnnet':
      model           = GnnNet( model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)
    elif params.method == 'matchingnet':
      model           = MatchingNet( model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
      if params.model == 'Conv4':
        feature_model = backbone.Conv4NP
      elif params.model == 'Conv6':
        feature_model = backbone.Conv6NP
      else:
        feature_model = model_dict[params.model]
      loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
      model           = RelationNet( feature_model, loss_type = loss_type, tf_path=params.tf_dir, **train_few_shot_params)
  else:
    raise ValueError('Unknown method')
  if torch.cuda.is_available():
    model = model.cuda()

  # load model
  start_epoch = params.start_epoch #0
  stop_epoch = params.stop_epoch #400
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  elif 'baseline' not in params.method:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method) # modify feature extractor paras
    model.feature.load_state_dict(state, strict=False)

  # training
  print('\n--- start the training ---')
  model = train(base_loader, val_loader,  model, start_epoch, stop_epoch, params)
