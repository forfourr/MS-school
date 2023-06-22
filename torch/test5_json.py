import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import AnyStr


class CustonDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.transform = transform

        #json불러오기
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        img_path = self.data[index]['filename']
        img_path = os.path.join("이미지 폴더",img_path)

        #image = Image.open(img_path)    #이미지 열기

        #json값 가지고오기
        bboxes = self.data[index]['ann']['bboxes']
        labels = self.data[index]['ann']['labels']
        #tensor 처리
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        #전처리 과정
        # if self.transform:
        #     image = self.transfrom(image)

        return img_path, {'boxes': bboxes, 'labels':labels}
    
    def __len__(self):
        return len(self.data)



if __name__ =='__main__':
    dataset = CustonDataset('data/test.json')
    
    for item in dataset:
        print(f"data {item}")