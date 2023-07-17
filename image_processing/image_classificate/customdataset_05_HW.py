import os
import cv2
import glob

from torch.utils.data import Dataset



class Customdataet(Dataset):
    def __init__(self, data_dir, transfrom=None):
        # data_dir = './data/food_dataset/train
        self.data_dir = glob.glob(os.path.join(datat_dir,"*","*.jpg"))
        self.transform = transfrom
        self.label_dict = self.create_label_dict()
    
    def __getitem__(self, index):
        pass

    def __len__(self,x):
        pass

PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'

if __name__ == '__main__':
    
    test = Customdataset(f"{PATH}/food_dataset/train")
    for i in test:
        pass