import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from noise_model import NoisyAutoEncoder

# transform
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
BATCH_SIZE=10
test_dataset = torchvision.datasets.MNIST(root='GAN/data', train=False,
                                           transform=transforms, download=False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size =BATCH_SIZE, shuffle=False )




# Model 
model = NoisyAutoEncoder()
model.load_state_dict(torch.load('GAN/data/noise_model.pt', map_location='cpu'))
model.eval()


for imgs,_ in test_loader:

    noise_factor = 0.2
    noisy_imgs = imgs + noise_factor*torch.randn(imgs.size())

    reconstructed_imgs = model(noisy_imgs)

    for j in range(BATCH_SIZE):
        fig, axes = plt.subplots(1,3,figsize=(15,5))

        # org img
        org_imgs = imgs[j].view(28,28)
        axes[0].imshow(org_imgs, cmap='gray')
        axes[0].set_title('org img')

        # noisy img
        noisy_img = noisy_imgs[j].view(28,28)
        axes[0].imshow(noisy_img, cmap='gray')
        axes[0].set_title('noisy img')

        # restructed img
        reconstructed_img = reconstructed_imgs[j].view(28,28)
        axes[0].imshow(reconstructed_img, cmap='gray')
        axes[0].set_title('reconstructed img')

        for ax in axes:
            ax.axis('off')  # 격자오프

        plt.show()