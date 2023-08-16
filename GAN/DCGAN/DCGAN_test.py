import os
import torch
import torchvision.transforms as transforms
from DCGAN_model import BaseDcganGenerator
from PIL import Image

nz = 100    # 잠자공간 벡터 크기
output_img_path = '/GAN/data/DCGAN/generated_imgs_256'
os.makedirs(output_img_path, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 생상자 모델 
netG = BaseDcganGenerator().to(device)
checkpoint_path = 'GAN/data/DCGAN/netG_epoch_45.pth'
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()


# 생성할 이미지 수
num_imgs = 10

# 생성한 이미지 저장 리스트
generated_imgs = []

# 잠재 공간 벡터 생성, 이미지 생성
with torch.no_grad():
    for _ in range(num_imgs):
        noise = torch.randn(1, nz, 1,1, device=device)
        fake = netG(noise)
        generated_imgs.append(fake.detach().cpu())

for i, image in enumerate(generated_imgs):
    image = torch.clamp(image, min=-1, max=1)
    image = (image+1)/2
    image = transforms.ToPILImage()(image.squeeze())
    image = image.resize((256,256), Image.BICUBIC)
    image.save(f"{output_img_path}generated_img_{i}.png")
