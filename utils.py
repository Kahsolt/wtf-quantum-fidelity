#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/10/24 

import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import torchvision.transforms as transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QCIFAR10Dataset(Dataset):

  def __init__(self):
    self.quantum_dataset = None

  def __len__(self):
    return len(self.quantum_dataset)

  def __getitem__(self, idx):
    return self.quantum_dataset[idx]


def reshape_norm_padding(x:Tensor) -> Tensor:
  PADDING_SIZE = 4096 - 3 * 32 * 32
  original_shape = x.shape
  x = x.reshape(original_shape[0] if x.dim() == 4 else 1, -1)
  x = F.normalize(x, p=2, dim=1)
  x = F.pad(x, (0, PADDING_SIZE), mode='constant', value=0)
  return x.flatten()    # (4096,)

def img_to_01(x:Tensor) -> Tensor:
  vmin, vmax = x.min(), x.max()
  x = (x - vmin) / (vmax - vmin)
  return x


# 赛方给出的实现
def get_fidelity_0(state_pred:Tensor, state_true:Tensor) -> float:
  # state_pred, state_true: (batch_size, 4096, 1)
  state_pred = state_pred.view(-1, 4096)
  state_true = state_true.view(-1, 4096)
  fidelity = torch.abs(torch.matmul(state_true.conj(), state_pred.T)) ** 2
  return fidelity.diag().mean().item()

# 我们简化后的等价实现
def get_fidelity_1(state_pred:Tensor, state_true:Tensor) -> float:
  state_pred = state_pred.view(-1, 4096).real
  state_true = state_true.view(-1, 4096).real
  fidelity = (state_pred * state_true).sum(-1)**2
  return fidelity.mean().item()

# 读取 dataset.pkl 并按 fid 排序
def get_dataset(args, sorted=True):
  with open(args.fp, 'rb') as file:
    test_dataset = pkl.load(file)

  dataset = []
  for x, vec_z in test_dataset:
    vec_x = reshape_norm_padding(x)
    fid0 = get_fidelity_0(vec_x, vec_z)
    fid1 = get_fidelity_1(vec_x, vec_z)
    assert abs(fid0 - fid1) < 1e-5
    z = vec_z.reshape(-1, 32, 32)[:3, ...]
    im_x = img_to_01(x).permute([1, 2, 0]).numpy()
    im_z = img_to_01(z).permute([1, 2, 0]).numpy()
    dataset.append((fid0, im_x, vec_x, im_z, vec_z))
  if sorted:
    dataset.sort(key=(lambda e: e[0]), reverse=True)
  fid_list = [e[0] for e in dataset]
  print('len(test_dataset):', len(dataset))
  print('  max(fid):', max(fid_list))
  print('  avg(fid):', sum(fid_list) / len(fid_list))
  print('  min(fid):', min(fid_list))
  return dataset

# Transfer im_x / im_z to tensor_x / tensor_z
def np_to_tensor(x:np.ndarray) -> Tensor:
  x = np.transpose(x, (2, 0, 1))
  tensor_x = torch.from_numpy(x)
  transform = transforms.Normalize(mean=mean, std=std)
  tensor_x = transform(tensor_x).unsqueeze(0)
  return tensor_x