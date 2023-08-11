import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from noise_model import NoisyAutoEncoder


BATCH_SIZE = 246
LR = 0.0025
NUM_EPOCHS = 50
LATENT_DIM = 20

transforms = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='GAN/data', train=True,
                                           transform=transforms, download=False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True )


# Add Noise
def add_noise(img, noise_factor=0.5):
    noisy_img = img + noise_factor*torch.rand_like(img)
    img = img.to('cuda')
    noisy_img = torch.clamp(noisy_img, -1, 1)

    return noisy_img


# Model 
model = NoisyAutoEncoder().to('cuda')
criterion = nn.MSELoss().to('cuda')
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


for epoch in range(NUM_EPOCHS):
    for i, (img, _) in enumerate(train_loader):
        noisy_img = add_noise(img, noise_factor=0.5)
        img = img.to('cuda')
        noisy_img = noisy_img.to('cuda')

        optimizer.zero_grad()

        outputs = model(noisy_img)

        loss = criterion(outputs.view(-1,784), img.view(-1,784))
        loss.backward()
        optimizer.step()

    print(f"EPOCH: [{epoch+1}/{NUM_EPOCHS}], loss: {loss.item():.3f}")

# save model
torch.save(model.state_dict(), 'GAN/data/noise_model.pt')