import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = glob.glob(os.path.join(data_dir,"*","*.jpg"))
        self.transform = transform
        #dict
        self.label_dict = {'Carpetweeds':0, 'Crabgrass':1, 'Eclipta':2, 'Goosegrass':3,
                            'Morningglory':4, 'Nutsedge':5, 'PalmerAmaranth':6, 'Prickly Sida':7,
                            'Purslane':8, 'Ragweed':9, 'Sicklepod':10, 'SpottedSpurge':11,
                            'SpurredAnoda':12, 'Swinecress':13, 'Waterhemp':14}

    def __getitem__(self, index):
        img_path =self.data_dir[index]
        image = Image.open(img_path)
        image = image.convert("RGB")
        label_name = img_path.split('\\')[1]
        label = self.label_dict[label_name]
        if self.transform is not None :
            image = self.transform(image)


        #img_path 추가해주는 이유 : test에서 어떤 데이터가 맞췄는디 cv2에서 보기위해
        return image ,label, img_path
    
    def __len__(self):
        return len(self.data_dir)
    
data_path = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'

# test = CustomDataset(f"{data_path}/new_plant/train",transform=None)
# for i in test:
#     print(i)