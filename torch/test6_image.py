import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import toTensorV2

class AlbumentationDataset(Dataset):
    def __itit__(self, file_paths, labels, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        file_path = self.file_paths[index]

        #read image
        image = cv2.read(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image= image)
            image = augmented['image']

        return image, label
        


    def __len__(self):
        return len(self.file_paths)
    


if __name__ =='__main__':
    photo_list = ['data/face.jpg',
                    'data/face1.jpg',
                    'data/face2.jpg']
    albumentaion_transform = A.Compose([
        A.Resize(256,256),
        A.RandomCrop(224,224),      #(256,256)사이즈 사진에서 (224,224)부분만 랜덤하게..?
        A.HorizontalFlip(),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]     #연구에 의한 값
        ),
        ToTensorV2()
    ])
    dataset = AlbumentationDataset(photo_list, [0,1,2],
                                   transforms=albumentaion_transform)
