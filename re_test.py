#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   re_test.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020-04-24 18:31   tangyubao      1.0         None
'''

# import lib
import torch
import os
import h5py
import torch.nn as nn
from methods import re_backbone
from methods.re_backbone import model_dict
from data.re_datamgr import SimpleREDataManager
from re_args import parse_args, get_best_file, get_assigned_file

from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.gnnnet import GnnNet
from methods.relationnet import RelationNet
import data.feature_loader as feat_loader
import random
import numpy as np

# extract and save image features
def save_features(model, data_loader, featurefile):
  f = h5py.File(featurefile, 'w')
  max_count = len(data_loader)*data_loader.batch_size
  all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
  all_feats=None
  count=0
  for i, (x,y,o) in enumerate(data_loader): # x[64,512]
    if (i % 10) == 0:
      print('    {:d}/{:d}'.format(i, len(data_loader)))
    if torch.cuda.is_available():
      x = x.cuda()
    feats = model(x) # 【64，460】
    if all_feats is None:
      all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
    all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
    all_labels[count:count+feats.size(0)] = y.cpu().numpy()
    count = count + feats.size(0)

  count_var = f.create_dataset('count', (1,), dtype='i')
  count_var[0] = count
  return f

# evaluate using features
def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15):
  class_list = cl_data_file.keys()
  select_class = random.sample(class_list,n_way)
  z_all  = []
  for cl in select_class:
    img_feat = cl_data_file[cl]
    perm_ids = np.random.permutation(len(img_feat)).tolist()
    z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )
  z_all = torch.from_numpy(np.array(z_all) )

  model.n_query = n_query
  scores  = model.set_forward(z_all, is_feature = True)
  pred = scores.data.cpu().numpy().argmax(axis = 1)
  y = np.repeat(range( n_way ), n_query )
  acc = np.mean(pred == y)*100
  return acc

# --- main ---
if __name__ == '__main__':

  # parse argument
  params = parse_args('test')
  print('Testing! {} shots on {} dataset with {} epochs of {}({})'.format(params.n_shot, params.dataset, params.save_epoch, params.name, params.method))
  remove_featurefile = True

  print('\nStage 1: saving features')
  # dataset
  print('  build dataset')

  split = params.split
  loadfile = os.path.join(params.data_dir, (params.dataset + '.json'))
  datamgr         = SimpleREDataManager(batch_size = 64, shuffle = params.shuffle)
  data_loader      = datamgr.get_data_loader(loadfile,params.max_length)

  print('  build feature encoder')
  checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
 # print(checkpoint_dir)
  if params.save_epoch != -1:
    modelfile   = get_assigned_file(checkpoint_dir,params.save_epoch)
  else:
    modelfile   = get_best_file(checkpoint_dir)
 # print(modelfile)
  if params.method in ['relationnet', 'relationnet_softmax']:
    if params.model == 'Conv4':
      model = re_backbone.Conv4NP()
    elif params.model == 'Conv6':
      model = re_backbone.Conv6NP()
    else:
      model = model_dict[params.model]( flatten = False )
  else:
    model = model_dict[params.model]()
  if torch.cuda.is_available():
    model = model.cuda()
  tmp = torch.load(modelfile)
  try:
    state = tmp['state']
 #   print(state.keys())
  except KeyError:
    state = tmp['model_state']
  except:
    raise
  state_keys = list(state.keys())
  for i, key in enumerate(state_keys):
    if "feature." in key and not 'gamma' in key and not 'beta' in key:
      newkey = key.replace("feature.","")
      state[newkey] = state.pop(key)
    else:
      state.pop(key)
 # print('-----------------------')
 # print(state.keys())
 # for name in model.state_dict():
 #   print(name)
 # print('--------------------------------------')
 # for name in state: 
 #   print(name)
  model.load_state_dict(state)
  model.eval()

  # save feature file
  print('  extract and save features...')
  if params.save_epoch != -1:
    featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_epoch)+ ".hdf5")
  else:
    featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")
  dirname = os.path.dirname(featurefile)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)
  # feature_path=os.path.join(split + ".hdf5")
  f = save_features(model, data_loader, featurefile)


  #f.close()

  print('\nStage 2: evaluate')
  acc_all = []
  iter_num = 1000
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) # 5，5
  # model
  print('  build metric-based model')
  if params.method == 'protonet':
    model = ProtoNet( model_dict[params.model], **few_shot_params,proto_attention=params.proto_attention,distance=params.proto_distance, common=params.proto_common)
  elif params.method == 'matchingnet':
    model = MatchingNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'gnnnet':
    model = GnnNet( model_dict[params.model], **few_shot_params)
  elif params.method in ['relationnet', 'relationnet_softmax']:
    if params.model == 'Conv4':
      feature_model = re_backbone.Conv4NP
    elif params.model == 'Conv6':
      feature_model = re_backbone.Conv6NP
    else:
      feature_model = model_dict[params.model]
    loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
    model = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
  else:
    raise ValueError('Unknown method')
  if torch.cuda.is_available():
    model = model.cuda()
  model.eval()

  # load model
  checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if params.save_epoch != -1:
    modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
  else:
    modelfile = get_best_file(checkpoint_dir)
  if modelfile is not None:
    tmp = torch.load(modelfile)
    #print(tmp['state'].keys())
    try:
      model.load_state_dict(tmp['state'])

#    except RuntimeError:
#      print('warning! RuntimeError when load_state_dict()!')
#      model.load_state_dict(tmp['state'], strict=False)
#    except KeyError:
#      for k in tmp['model_state']:   ##### revise latter
#        if 'running' in k:
#          tmp['model_state'][k] = tmp['model_state'][k].squeeze()
#      model.load_state_dict(tmp['model_state'], strict=False)
    except:
      raise

  # load feature file
  print('  load saved feature file')
  cl_data_file = feat_loader.init_loader(f)

  # start evaluate
  print('  evaluate')
  for i in range(iter_num):
    acc = feature_evaluation(cl_data_file, model, n_query=5, **few_shot_params)
    acc_all.append(acc)

  # statics
  print('  get statics')
  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

  # remove feature files [optional]
  if remove_featurefile:
    os.remove(featurefile)


