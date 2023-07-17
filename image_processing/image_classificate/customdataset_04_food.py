import os
import cv2
import glob

from torch.utils.data import Dataset

class Customdataset(Dataset):
    def __init__(self, datat_dir, transfrom=None):
        # data_dir = './data/food_dataset/train
        self.data_dir = glob.glob(os.path.join(datat_dir,"*","*.jpg"))
        self.transform = transfrom
        self.label_dict = self.create_label_dict()

    # 폴더명에서 label 갖고 오기
    def create_label_dict(self):
        label_dict = {}
        for filepath in self.data_dir:
            label = os.path.basename(os.path.dirname(filepath))
            if label not in label_dict:
                label_dict[label] = len(label_dict)

        return label_dict

    def __getitem__(self, index):
        img_filepath = self.data_dir[index]
        img = cv2.imread(img_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = os.path.basename(os.path.join(img_filepath))
        label_idx = self.label_dict[label]

        # augmentation
        if self.transform is not None:
            image = self.transform(image=img)['image']
        
        return image, label_idx

    def __len__(self):
        pass

PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'

if __name__ == '__main__':
    
    test = Customdataset(f"{PATH}/food_dataset/train")
    for i in test:
        pass