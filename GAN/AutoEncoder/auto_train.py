## Auto Encoder example with MNIST data

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from auto_model import AutoEncoder

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
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True )


# Model 
model = AutoEncoder().to('cuda')
criterion = nn.MSELoss().to('cuda')
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


## train
for epoch in range(NUM_EPOCH):
    start_time = time.time()

    for data in train_loader:
        img,_ = data
        img = img.to('cuda')
        img= img.view(img.size(0), -1)  # MNIST 이미지는 28x28 픽셀의 2D -> 1D

        optimizer.zero_grad()

        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()

    end_time = time.time()

    epoch_time = end_time - start_time
    print(f"EPOCH: [{epoch+1}/{NUM_EPOCH}], Loss: {loss.item():.4f}, time: {epoch_time} sec")

# save trained model
torch.save(model.state_dict(), 'autoencoder_model.pt')



