import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

def imgaug_tranform(image: torch.Tensor):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Multiply((0.8, 1.2))
    ])
    image_np = image.permute(1,2,0).numpy()
    image_aug = seq(image = image_np)
    image_aug_copy = image_aug.copy()   #바로 텐서로 바꾸면 주소가 공유되기때문에
    image_aug_tensor = torch.from_numpy(image_aug_copy).permute(2,0,1)
    return image_aug_tensor


def transform_data(image):
    tensor = transforms.ToTensor()(image)
    transformed_tensor = imgaug_tranform(tensor)
    return transformed_tensor

#CIFAR-10데이터 로드
train_dataset = torchvision.datasets.CIFAR10(root='/data', train=True,
                                             download=True,
                                             transform=transform_data)  #콜백함수
#데이터 로더 설정
batch_size = 4
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

for images,labels in train_dataloader:
    fig, axes = plt.subplots(1, batch_size, figsize=(12,4))

    #배치 사이즈 크기만큼 반복
    for i in range(batch_size):
        image = images[i].permute(1,2,0).numpy()
        axes[i].imshow(image)