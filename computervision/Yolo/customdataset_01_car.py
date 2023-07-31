import os
import cv2
import glob
import pandas as pd
import numpy as np
import csv
import torch
from torch.utils.data import Dataset


'''
differnet shape or type (image, bounding boxes) -> collate function
DataLoader 의 역할을 해준다.
'''

def collate_fn(batch):
    '''
    batch = [
        (image, box, label),
        (....)
    ]
    '''
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    print(f"zip*batch: {zip(*batch)},\n tuple:{tuple(zip(*batch))} ")

    # img list -> torch.stack
    # 여러 개의 텐서를 하나의 새로운 텐서로 결합
    images = torch.stack(images,1)
    targets = []    # >> targets_boxes, target_labels

    for i in range(len(targets_boxes)):
        target = {
            'boxes': targets_boxes[i],
            'labels': targets_labels[i]
        }
        targets.append(target)
        # target정보가 담긴
    return images, targets

class Customdataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.img_path = sorted(glob.glob(root+'/*.png'))
        # self.imags = sorted(glob.glob(os.path.join(root, '*.png')))

        # train을 할 때
        if train:
            self.boxes = sorted(glob.glob(os.path.join(root, '*.txt')))

    # boxes의 정보 가져오기
    def parse_boxes(self, box_path):
        # read txt file
        with open(box_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        boxes = []
        labels = []
        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = values[0]
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))),\
                            int(round(max(values[4], values[6], values[8])))
            boxes.append([x_min, y_min,x_max,y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32),\
                torch.tensor(labels, dtype=torch.int64)

            
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img.shape = (1040, 1920, 3)
        img /= 255.0    # >> normalization
        height, width = img.shape[0], img.shape[1]      # height: 1040, widht:1920

        # train 일 때
        if self.train:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1     # background = 0

            if self.transforms is not None:
                transformed = self.transforms(image= img, bboxes = boxes, labels=labels)
                img, boxes, labels= transformed['image'], transformed['bboxes'], transformed['labels']

            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
        # test/valid 일 때
        else:
            if self.transforms is not None:
                transformed = self.transforms(image= img)
                img = transformed['image']
            file_name = img_path.split('/')[-1]
            return file_name, img, width, height


    def __len__(self):
        return len(self.img_path)

# if __name__ == '__main__':
#     test = Customdataset('computervision/data/car_load_dataset/train', train=True, transforms=None)
#     for i in test:
#         print(i)
