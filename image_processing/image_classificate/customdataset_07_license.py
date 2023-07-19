import os
import cv2
import glob
import pandas as pd

from torch.utils.data import Dataset
    
#PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'


class Customdataset(Dataset):
    def __init__(self,data_dir, transfrom=None):
        self.data_dir = glob.glob(os.path.join(data_dir,"*","*.jpg"))
        self.transform = transfrom

        self.label_dict = self.make_label_dict()
    
    def make_label_dict(self):
        label_dict = {}
        for filepath in self.data_dir:
            label = os.path.basename(os.path.dirname(filepath))
            if label not in label_dict:
                label_dict[label] = len(label_dict)
        return label_dict



    def __getitem__(self,index):
        img_path = self.data_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label =os.path.basename(os.path.dirname(img_path))
        label_idx = self.label_dict[label]

        if self.transform is not None:
            img = self.transform(image = img)['image']

        # img shape (128, 224, 3)

        return img, label_idx

    def __len__(self):
        return self.data_dir




# if __name__ == "__main__":
#     test = Customdataset(f"{PATH}/license_dataset/train")
#     for i in test:
#         pass
