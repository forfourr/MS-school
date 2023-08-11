import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


BATCH_SIZE = 128

transforms = transforms.Compose([
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='GAN/data',train=True,
                                           transform=transforms, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)


# Add Noise
def add_noise(img, noice_factor=0.5):
    noisy_img = img + noice_factor*torch.rand_like(img)
    noisy_img = torch.clamp(noisy_img, -1, 1)

    return noisy_img


for img, _ in train_loader:
    noisy_imgs = add_noise(img)
    break

import matplotlib.pyplot as plt

fig,axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(np.transpose(torchvision.utils.make_grid(img[:8],padding=2, normalize=True),(1,2,0)))
axes[0].set_title('origin img')
axes[0].axis('off')
'''
8개 이미지 선택 -> torchvision.utils.make_gird를 통해 -> img grid 생성
nosiy img = [128,1,,28,28]
-> (1,2,0)으로 차원 재배열 -> RGB이미지
'''

axes[1].imshow(np.transpose(torchvision.utils.make_grid(img[:8],padding=2, normalize=True),(1,2,0)))
axes[1].set_title('nosiy img')
axes[0].axis('off')

plt.show()