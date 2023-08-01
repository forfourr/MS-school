import os
import cv2
import glob
import pandas as pd
import csv
from torch.utils.data import Dataset

'''
train/val data 폴더 안에 있는 이미지 이름
-> train_labels/val_labels .csv 파일 안에 idx와 label 매칭 후 dict 생성
-> calss_list.txt 안에 label_idx와 매칭
'''

class Customdataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = glob.glob(os.path.join(data_dir,"*","*.jpg"))
        self.transform = transform
        self.mode = mode

        # label dict 생성
        self.label_dict = self.make_label_dict()


    def make_label_dict(self):
        labels_dict={}
        txt_file_path = f"{PATH}/08_food_HW_data/class_list.txt"

        with open(txt_file_path, 'r') as file:
            for idx, line in enumerate(file):
                label = line.strip().split(' ')[1]  # 줄 바꿈 문자를 제거하여 레이블 추출
                labels_dict[label] = idx
   
        return labels_dict
    

    def __getitem__(self, index):
        img_path = self.data_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # label 구하기
        #img_name = os.path.basename(img_path)
        label = os.path.basename(os.path.dirname(img_path))
        label_idx = self.label_dict[label]    

        #augmentation
        if self.transform is not None:
            img = self.transform(image = img)['image']
        return img, label_idx

    def __len__(self):
        return len(self.data_dir)


PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
#PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'


# if __name__ =="__main__":
#     test = Customdataset(f"{PATH}/08_food_HW_data/valid_new", 'valid')
    
#     print(test.label_dict)