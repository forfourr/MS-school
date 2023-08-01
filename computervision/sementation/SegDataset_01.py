from typing import Any, Tuple
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor
from skimage.segmentation import mark_boundaries
import numpy as np
import os
import cv2

# Dataset의 원시데이터 상속이 아닌, datasets의 VOCSegmentation 함수 상속
# len함수의 재정의가 필요 없다
# __init__에서 super init만 불러움

class customVOCSegmentation(VOCSegmentation):
    # Customdataset 생성자
    def __init__(self, root, mode = 'train', transforms = None):
        self.root = root
        super().__init__(root=root, image_set=mode, download=False, transforms=transforms)
        # VOCSegmentation.__init__()과 같다
    
    def __getitem__(self, idx):
        # 부모 클래스에서 호출해온 것
        img = self.images[idx]
        img = cv2.imread(img)
        mask = self.masks[idx]
        mask = cv2.imread(mask)
        
        # self.transforms is not None 이랑 같음
        if self.transforms:
            aug = self.transforms(image= img, mask= mask)
            img = aug['image']
            mask = aug['mask']



        return img, mask

    # 클래스 선언과 동시에 다운, 이미 다운되어 있으면 Fasle 반환
    # def check_if_path_exist(self):
    #     return False if os.path.exists(self.root) else True






# if __name__ == '__main__':
#     voc = customVOCSegmentation('computervision/sementation/data')
#     for i in voc:
#         img,mask = i
#         summary = cv2.copyTo(img,mask)
#         marked = cv2.addWeighted(img, 0.5,summary,0.5,0)


#         cv2.imshow('with mask',marked)
#         key = cv2.waitKey()
#         cv2.destroyAllWindows()
#         if key == ord('q'):
#             break