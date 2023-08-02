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



    '''
    transform을 위해서 저렇게 2개 요소로만 이루어진 keypoint를 생성하는 이유
    데이터 albumentation에서는 2차원 계산이 적용되어서 저런 작업을 해준거맞나요? 
    다시 3차원으로 변경해준 이유는 학습시에는 3차원 텐서형태가 필요해서인거구요
    '''
    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'images',self.img_files[index])
        anno_path = os.path.join(self.root, 'annotations', self.anno_files[index])
        

    def __len__(self):
        pass


if __name__ =='__main__':
    test = keypointDataset('computervision/keypointRCNN/keypoint_dataset/train')