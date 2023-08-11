## Auto Encoder example with MNIST data

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Hyperparameter
BATCH_SIZE = 248
LR = 0.001
NUM_EPOCH = 100

# transformer
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))         # gray scale -> channel =1
])

# dataset
train_dataset = torchvision.datasets.MNIST(root='GAN/data', train=True, transform=transforms, download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffles=True )