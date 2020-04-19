# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from methods.meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support, tf_path=None, hidden_size=230):
    super(ProtoNet, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)
    self.loss_fn = nn.CrossEntropyLoss()
    self.method = 'ProtoNet'
    self.hidden_size = hidden_size
    self.common_gain = nn.Parameter(torch.ones(hidden_size, dtype=torch.float))
  def reset_modules(self):
    return

  def set_forward(self,x,is_feature=False):
    z_support, z_query  = self.parse_feature(x,is_feature) # [5,5,512], [5,16,512]
    z_support   = z_support.contiguous()
    z_support   = z_support.view(self.n_way, self.n_support, -1 )
    z_proto     = z_support.float().mean(1) #the shape of z is [n_data, n_dim] [N,K,D]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ).float()
    common_gain = self.common_gain / self.common_gain.sum() * self.hidden_size
    #differ_gain = self.differ_gain / self.differ_gain.sum() * self.hidden_size
    class_common = z_proto * common_gain  # (N, D)

    support_differ = abs((z_support - class_common.unsqueeze(1)))  # (N, K, D)
    class_differ = torch.mean(support_differ, 1)  # (N, D)
    class_differ = class_differ.mean(1) # (N)
    class_weight = F.softmax(class_differ / 15, dim=0) * self.n_way
    #class_differ = class_differ * differ_gain
    #class_differ = class_differ.mean(2)  # (B, N)  # with class_differ increase, the class may have less confidence
    z_proto = z_proto.transpose(0, 1) * class_weight  #(D,N)
    z_proto = z_proto.transpose(0, 1)  # (N, D)
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim] [5,512]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ) #[80,512]

    dists = euclidean_dist(z_query, z_proto) #[80,5]
    scores = -dists
    return scores

  def get_distance(self,x,is_feature = False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
    return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

  def set_forward_loss(self, x): # x [5,21,3,224,224]
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query )) #[80]
    if torch.cuda.is_available():
      y_query = y_query.cuda()
    scores = self.set_forward(x) #[80,5]
    loss = self.loss_fn(scores, y_query)
    return scores, loss


def euclidean_dist( x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

class Proto(nn.Module):  ## only for test!!
  def __init__(self,hidden_size=230):
    super(Proto, self).__init__()
    self.loss_fn = nn.CrossEntropyLoss()
    self.method = 'ProtoNet'
    self.hidden_size = hidden_size
    self.common_gain = nn.Parameter(torch.ones(hidden_size, dtype=torch.float))
  def reset_modules(self):
    return

  def forward(self,x,is_feature=False):
    #z_support, z_query  = self.parse_feature(x,is_feature)
    #z_support   = z_support.contiguous()
    z_support = torch.arange(6*230).float()
    z_query = torch.arange(6*230).float()

    z_proto     = z_support.view(2, 3, -1 ) #the shape of z is [n_data, n_dim] [N,K,D]
    z_support = z_support.view(2,3,-1)
    #z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

    common_gain = self.common_gain / self.common_gain.sum() * self.hidden_size
    #differ_gain = self.differ_gain / self.differ_gain.sum() * self.hidden_size
    class_common = torch.mean(z_proto * common_gain, 1)  # (N, D)

    support_differ = abs((z_support - class_common.unsqueeze(1)))  # (N, K, D)
    class_differ = torch.mean(support_differ, 1)  # (N, D)
    class_differ = torch.sum(class_differ, 1)  # (N)
    #
    #class_weight = F.softmax(class_differ / 15, dim=0) * self.n_way
    #class_differ = class_differ * differ_gain
    #class_differ = class_differ.mean(2)  # (B, N)  # with class_differ increase, the class may have less confidence
    z_proto = z_proto.transpose(0, 1) * class_weight  #(D,N)
    z_proto = z_proto.transpose(0, 1)  # (N, D)

    dists = euclidean_dist(z_query, z_proto)
    scores = -dists
    return scores

  def get_distance(self,x,is_feature = False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
    return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
    if torch.cuda.is_available():
      y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss


if __name__=='__main__':
  net = Proto()
  x = torch.arange(2300)
  y = net(x)
  print(y)
