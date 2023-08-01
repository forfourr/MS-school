import json
import os
from torch.utils.data import Dataset

class keypointDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.img_files = sorted(os.listdir(os.path.join(self.root, 'images')))
        self.anno_files = sorted(os.listdir(os.path.join(self.root,'annotations')))
        print(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'images',)

    def __len__(self):
        pass


if __name__ =='__main__':
    test = keypointDataset('computervision/keypointRCNN/keypoint_dataset/train')
