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
        self.label_dict = {"Abstract" : 0 , "Cubist" : 1, "Expressionist" : 2,
                   "Impressionist" : 3, "Landscape" : 4, "Pop Art":5,
                   "Portrait" : 6, "Realist" :7, "Still Life" : 8,
                   "Surrealist" : 9}

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

test = CustomDataset(f"{data_path}/paintings/train",transform=None)
