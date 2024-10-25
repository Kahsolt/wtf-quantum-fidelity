#!/usr/bin/env python3
# Author: FlowerWater1019
# Create Time: 2024/10/25 

# 训练 im_x -> fid

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

import random
from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision.models as M
import numpy as np
import matplotlib.pyplot as plt

from utils import *

seed = 114514
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = lambda x: sum(x) / len(x)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def get_model(name:str) -> nn.Module:
  if name == 'vgg11':
    model = M.vgg11(pretrained=True)
    for param in model.parameters():
      param.requires_grad = True

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    model.features[0].requires_grad_(True)
    model.classifier[-1].requires_grad_(True)

  elif name == 'resnet18':
    model = M.resnet18(pretrained=True)
    for param in model.parameters():
      param.requires_grad = True

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    model.conv1.requires_grad_(True)
    model.bn1.requires_grad_(True)
    model.fc.requires_grad_(True)

  return model


class MyDataset(Dataset):
  def __init__(self, samples):
    super().__init__()
    self.samples = samples
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, idx):
    return self.samples[idx]


def run(args):
  raw_dataset = get_dataset(args)
  fid_list = [e[0] for e in raw_dataset]
  if not 'plot fid hist':   # 已经呈现为较为对称的单峰分布，只需要z-score规范化
    plt.hist(fid_list, bins=50)
    plt.show()
  if 'norm fid target':
    fid_avg = np.mean(fid_list)
    fid_std = np.std(fid_list)
    print('avg(fid):', fid_avg)   # 0.9645763841867447
    print('std(fid):', fid_std)   # 0.018195116032846164
    dataset = [(e[1].transpose(2, 0, 1), (e[0] - fid_avg) / fid_std) for e in raw_dataset]   # [(X, Y)]
  random.shuffle(dataset)

  cp = int(len(dataset) * args.split_ratio)
  trainloader = DataLoader(MyDataset(dataset[:cp]), batch_size=args.batch_size, pin_memory=True, shuffle=True,  drop_last=True)
  validloader = DataLoader(MyDataset(dataset[cp:]), batch_size=args.batch_size, pin_memory=True, shuffle=False, drop_last=False)

  model = get_model(args.model).to(device)
  optim = Adam(model.parameters(), lr=1e-3)
  train_loss_list = []
  valid_loss_list = []
  for epoch in range(args.epoch):
    print(f'Epoch {epoch + 1}/{args.epoch}', end=' ')

    loss_list = []
    with torch.enable_grad():
      model.train()
      for X, Y in trainloader:
        X, Y = X.to(device), Y.float().to(device)

        optim.zero_grad()
        output = model(X).squeeze(-1)
        loss = F.mse_loss(output, Y)
        loss.backward()
        optim.step()
        loss_list.append(loss.item())

    train_loss_list.append(mean(loss_list))
    print(f'[Train] loss: {train_loss_list[-1]:.7f}', end='  ')

    loss_list = []
    with torch.inference_mode():
      model.eval()
      for X, Y in validloader:
        X, Y = X.to(device), Y.float().to(device)

        output = model(X).squeeze(-1)
        loss = F.mse_loss(output, Y)
        loss_list.append(loss.item())

    valid_loss_list.append(mean(loss_list))
    print(f'[Valid] loss: {valid_loss_list[-1]:.7f}')

  print('===== [pred] =====')
  testloader = DataLoader(MyDataset(dataset), batch_size=1, pin_memory=False, shuffle=False, drop_last=False)
  err_list = []
  with torch.inference_mode():
    model.eval().to('cpu')
    for idx, (X, Y) in enumerate(testloader):
      X, Y = X, Y.float()
      output = model(X).squeeze(-1)
      fid_pred  = Y     .item() * fid_std + fid_avg   # denorm fid
      fid_truth = output.item() * fid_std + fid_avg
      err = abs(fid_pred - fid_truth)
      err_list.append(err)
      print(f'[sample-{idx+1}] error: {err:.7f}')
  print('===== [pred] =====')

  '''
  [resnet18]
  >> theoretical fidelity estimate error: ±0.004602993408407657
  >> actual fidelity estimate error: 0.0028561596719686704
  [vgg11]
  >> theoretical fidelity estimate error: ±0.018334930729303735
  >> actual fidelity estimate error: 0.0140461407758442
  '''
  print(f'>> theoretical fidelity estimate error: ±{valid_loss_list[-1] * fid_std}')
  print(f'>> actual fidelity estimate error: {mean(err_list)}')

  plt.clf()
  plt.plot(train_loss_list, 'b', label='train_loss')
  plt.plot(valid_loss_list, 'r', label='valid_loss')
  plt.legend()
  plt.suptitle('loss')
  plt.tight_layout()
  plt.savefig('./img/train-resnet18.png', dpi=400)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=['vgg11', 'resnet18'], help='model')
  parser.add_argument('-F', '--fp', default='./data/test_dataset.pkl', help='path to encoded test_dataset.pkl')
  parser.add_argument('-E', '--epoch',      default=30, type=int, help='epoch')
  parser.add_argument('-B', '--batch_size', default=16, type=int, help='batch size')
  parser.add_argument('--split_ratio',      default=0.8, type=float, help='split ratio')
  args = parser.parse_args()
  
  run(args)
