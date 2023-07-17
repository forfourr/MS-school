import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = glob.glob(os.path.join(data_dir,"*","*.png"))
        self.transform = transform
        #dict
        self.label_dict = {'Mel':0,
                           "STFT":1,
                           "waveshow":2}

    def __getitem__(self, index):
        img_path =self.data_dir[index]
        image = Image.open(img_path)
        image = image.convert("RGB")
        label_name = img_path.split('\\')[1]
        label = self.label_dict[label_name]
        if self.transform is not None :
            image = self.transform(image)

        return image ,label, img_path
    
    def __len__(self):
        return len(self.data_dir)
    
data_path = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
test = CustomDataset(f"{data_path}/sound/train",transform=None)
