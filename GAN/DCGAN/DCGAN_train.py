import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dast
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from DCGAN_model import BaseDcganGenerator, BaseDcganDiscriminator
from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 초기화
# 가중치 난수 초기화 -> 속도, 모델학습 개선
def weight_init(m):
    classname = m.__class__.__name__

    # conv열 가중치 초기화
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorㅡ') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, val=0)



def main():
    dataset = dast.ImageFolder(root = data_root,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    real_batch = next(iter(data_loader))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('traning data Img view')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                            padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


    # 생성자 생성
    netG = BaseDcganGenerator().to(device)
    # 가중치 초기화
    netG.apply(weight_init)
    #rint(netG)

    # 구분자 생성
    netD = BaseDcganDiscriminator().to(device)
    # 모델초기화: 가중치 초기화
    netD.apply(weight_init)
    #print(netD)
    

    # loss function
    criterion = nn.BCELoss()
    # Generator input noise
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)


    # 생성자의 학습 상태 확인 할 공간 벡터
    fixed_noise = torch.randn(128, nz, 1, 1, device = device)

    # set training label
    real_label = 1.
    fake_label = 0.

    criterion = nn.BCELoss()
    optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1,0.999))
    optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1,0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    output_img_path = 'GAN/data/DCGAN/generated_imgs'
    os.makedirs(output_img_path,exist_ok=True)

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader, 0):


            ##### 신경망 업데이트 #####
            netD.zero_grad()    # 판별자 그래디언트 초기화

            real_gpu = data[0].to(device)   # 데이터를 CPU로 올림
            b_size = real_gpu.size(0)       # 현재 bathsize (128, 3, 64 64)
            # 판별자 실제인지 가짜인지 알려줌
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # 실제 데이터 -> 판별 모델 -> 출력(1차원)
            output = netD(real_gpu).view(-1)    

            loss_D = criterion(output, label)
            loss_D.backward()
            D_x = output.mean().item()

            #### 가짜 데이터로 학습 ####
            noise = torch.randn(b_size, nz, 1,1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            # D로 데이터 진위 판별
            output = netD(fake.detach()).view(-1)

            loss_D_fake = criterion(output, label)
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()

            # 가짜 이미지, 진짜 이미지 -> 손실함수 더함
            error_D = loss_D + loss_D_fake
            optimizerD.step()


            #### 신경망 업데이트 #####
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)
            loss_G = criterion(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # 훈련 상태를 출력합니다
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(data_loader),
                         error_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(loss_G.item())
            D_losses.append(error_D.item())

            # fixed noise -> 6 images append
            if (iters % 500 == 0 ) or ((epoch == num_epochs -1) and (i == len(data_loader)-1)) :
                with torch.no_grad() :
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # model save
            if epoch % 5 == 0 :
                os.makedirs("./DCGAN_model_weight/", exist_ok=True)
                torch.save(netG.state_dict(), f"./DCGAN_model_weight/netG_epoch_{epoch}.pth")
                torch.save(netD.state_dict(), f"./DCGAN_model_weight/netD_epoch_{epoch}.pth")

            # epoch 5
            if (epoch + 1) % 5 == 0:
                vutils.save_image(img_list[-1], f"{output_img_path}/fake_image_epoch_{epoch}.png")

            iters +=1







if __name__ == '__main__':
    main()