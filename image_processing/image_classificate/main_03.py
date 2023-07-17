import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_v2_s
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset_03_plant import CustomDataset
import torch.optim as optim
from torch.optim import AdamW
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########### ex1 #########
# model = efficientnet_v2_s(pretrained=True)
# in_features_ = 1280
# model.classifier[1] = nn.Linear(in_features_, 6)
# model.to(device)

# # aug
# train_transforms = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.RandomHorizontalFlip(p=0.4),
#     transforms.RandomVerticalFlip(p=0.4),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(),
#     transforms.RandAugment(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))
# ])

# val_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
# ])

########### ex2 #########
model = efficientnet_v2_s(pretrained=True)
in_features_ = 1280
model.classifier[1] = nn.Linear(in_features_, 15)
model.to(device)

# aug
train_transforms = transforms.Compose([
    transforms.CenterCrop((244,244)),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomRotation(degrees=15),
    transforms.RandAugment(),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.CenterCrop((244, 244)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])