import os
import cv2
import glob
import pandas as pd
import numpy as np
import csv
import torch
from torch.utils.data import Dataset

train_dir = 'faster-RCNN/Uno_cards_dataset/train'
valid_dir = 'faster-RCNN/Uno_cards_dataset/valid'


def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))

    images = torch.stack(images, 1)
    targets = []

    for i in range(len(targets_boxes)):
        target = {
            'boxes': targets_boxes[i],
            'labels': targets_labels[i]
        }
        targets.append(target)

    return images, targets


class Customdataset(Dataset):
    def __init__(self, dir, train=True, transforms=None):
        self.dir = dir
        self.train = train
        self.transforms = transforms
        self.img_path = sorted(glob.glob(dir + '/*.jpg'))


    def read_csv(self, img_name):
        anno = os.path.join(self.dir,'_annotations.csv')

        with open(anno, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if self.train:
            boxes = []
            labels = []
            # line:  filename,width,height,class,xmin,ymin,xmax,ymax
            for line in lines:
                values = line.strip().split(',')
                if values[0] == img_name:
                    class_id = float(values[3])
                    x_min, y_min = int(round(float(values[4]))), int(round(float(values[5])))
                    x_max, y_max = int(round(float(values[6]))), int(round(float(values[7])))
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
            return torch.tensor(boxes, dtype=torch.float32),\
                    torch.tensor(labels, dtype=torch.int64)
        
        else:
            for line in lines:
                values = line.strip().split(',')
                if values[0] == img_name:
                    width, height = float(values[1]), float(values[2])
                    return width, height

        


    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img_name =  os.path.basename(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        # train
        if self.train:
            boxes, labels = self.read_csv(img_name)
            labels +=1

            if self.transforms is not None:
                transformed = self.transforms(image= img, bboxes = boxes, labels=labels)
                img, boxes, labels= transformed['image'], transformed['bboxes'], transformed['labels']

            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
        # test/valid 일 때
        else:
            width, height = self.read_csv(img_name)

            if self.transforms is not None:
                transformed = self.transforms(image= img)
                img = transformed['image']

            return img_name, img, width, height

    def len(self):
        return len(self.dir)

# if __name__ == '__main__':
#     test = Customdataset(valid_dir,train=False)
#     for i in test:
#         print(i)