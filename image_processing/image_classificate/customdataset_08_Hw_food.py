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
        self.label_dict, self.label_dict2 = self.make_label_dict()

        # 파일 별로 label 찾기
        self.math_dict = self.match_label()

    def make_label_dict(self):
        labels_dict={}
        labels_dict2={}
        txt_file_path = f"{PATH}/08_food_HW_data/class_list.txt"

        with open(txt_file_path, 'r') as file:
            for idx, line in enumerate(file):
                label = line.strip().split(' ')[1]  # 줄 바꿈 문자를 제거하여 레이블 추출
                labels_dict[label] = idx
                labels_dict2[idx] = label
                
        return labels_dict, labels_dict2
    
    
    def match_label(self):
        match_dict = {}
        csv_path = f"{PATH}/08_food_HW_data/{self.mode}_labels.csv"
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)    #첫번째 헤더 제외
            for row in reader:
                img_name, label = row
                match_dict[img_name] = int(label)
        return match_dict

    def __getitem__(self, index):
        img_path = self.data_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_name = os.path.basename(img_path)
        label = self.math_dict[img_name]

        print(img_name, label, self.label_dict2[label])

        #augmentation
        if self.transform is not None:
            img = self.transform(image = img)['image']
        return img

    def __len__(self):
        return len(self.data_dir)


PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
#PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'


if __name__ =="__main__":
    test = Customdataset(f"{PATH}/08_food_HW_data/train_set", 'train')
    for i  in test:
        pass