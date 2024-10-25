from argparse import ArgumentParser
from utils import *
import torchvision.models as models
import torch.optim as optim


PRETRAINED_MODEL = {
  'resnet18': models.resnet18,
  'vgg11': models.vgg11,
}

def finetune(args):
  dataset = get_dataset(args, sorted=False)
  
  model = PRETRAINED_MODEL[args.model](pretrained=True)
  if args.model == 'resnet18':
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
      nn.Linear(num_features, 1),
      nn.Sigmoid()
    )
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=3, bias=False)
  else:
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
      nn.Linear(num_features, 1),
      nn.Sigmoid()
    )
    
  model.to(device)
  
  for param in model.parameters():
    param.requires_grad = True
    
  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  
  # 这里取前450个作为微调样本, 后50个作为测试样本
  finetune_num = 450
  
  # Finetune
  for epoch in range(args.epoch):
    print(f'Epoch {epoch}/{args.epoch - 1}')
    print('-' * 10)
    
    running_loss = 0.0
    for idx in range(finetune_num):
      fid, im_x, _, _, _ = dataset[idx]  
      fid = torch.tensor([fid], dtype=torch.float32).to(device)
      tensor_x = np_to_tensor(im_x).to(device)
      
      optimizer.zero_grad()
      with torch.set_grad_enabled(True):
        output = model(tensor_x).squeeze(0)
        loss = criterion(output, fid)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
      if idx % 30 == 0:
        print(f'idx: {idx}, loss: {running_loss / 10}')
        running_loss = 0.0
    
    for idx in range(finetune_num, len(dataset)):
      fid, im_x, _, _, _ = dataset[idx]
      fid = torch.tensor([fid], dtype=torch.float32).to(device)
      tensor_x = np_to_tensor(im_x).to(device)
      
      with torch.no_grad():
        output = model(tensor_x).squeeze(0)
        loss = criterion(output, fid)
      
      print(f'idx: {idx}, fid: {fid.item()}, pred_fid: {output.item()}, loss: {loss.item()}')
    
    
if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=PRETRAINED_MODEL.keys(), help='pretrained model')
  parser.add_argument('-F', '--fp', default='./data/test_dataset.pkl', help='path to encode test_dataset.pkl')
  parser.add_argument('-E', '--epoch', default=5, type=int, help='number of samples to finetune')
  args = parser.parse_args()
  
  finetune(args)