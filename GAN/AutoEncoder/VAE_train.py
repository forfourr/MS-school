import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

from VAE_model import VAE

BATCH_SIZE = 246
LR = 0.0025

NUM_EPOCHS = 100

# transfrom
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,),(1,))
])

train_dataset = torchvision.datasets.MNIST(root='GAN/data', train=True,
                                           transform=transforms, download=False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size =BATCH_SIZE, shuffle=True )

model = VAE()
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    for i, (imgs,_) in enumerate(train_loader):
        imgs = imgs.view(imgs.size(0), -1)
        optimizer.zero_grad()

        recon_img, mu, logvar = model(imgs)

        reconstruction_loss = criterion(recon_img, imgs) / BATCH_SIZE

        kl_divergence = -0.5*torch.sum(1+ logvar - mu.pow(2) - logvar.exp())/ BATCH_SIZE

        loss = reconstruction_loss + kl_divergence

        loss.backward()
        optimizer.step()

    print(f"Epoch: [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "GAN/data/vae_model.pt")