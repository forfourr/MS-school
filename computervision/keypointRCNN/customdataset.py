import numpy as np
import os
import torch
import cv2
import json
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class keypointDataset(Dataset):
    def __init__(self, root, transforms=None, demo=False):
        self.root = root
        self.demo = demo
        self.transforms = transforms

        self.img_files = sorted(os.listdir(os.path.join(self.root, 'images')))
        self.anno_files = sorted(os.listdir(os.path.join(self.root,'annotations')))


    '''
    transform을 위해서 저렇게 2개 요소로만 이루어진 keypoint를 생성하는 이유
    데이터 albumentation에서는 2차원 계산이 적용되어서 저런 작업을 해준거맞나요? 
    다시 3차원으로 변경해준 이유는 학습시에는 3차원 텐서형태가 필요해서인거구요
    '''
    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'images',self.img_files[index])
        anno_path = os.path.join(self.root, 'annotations', self.anno_files[index])
        
        img_org = cv2.imread(img_path)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)


        ### JSON 파일 읽기 ###
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            bboxes_org = data['bboxes']
            keypoints_org = data['keypoints']

            ## 모든 bbox에 대해 'Glue tube' 라벨 설정
            bboxes_labels_org = ['Glue tube' for _ in bboxes_org]


        if self.transforms:
            # keypoints flatten
            # keypoints =  [[[1019, 487, 1], [1432, 404, 1]], [[861, 534, 1], [392, 666, 1]]]
            # 2차원 평면이기 때문에 마지막 1 제거
            # el[0:2] = [1019,487]
            keypoints_org_flattened = [el[0:2] for kp in keypoints_org for el in kp]
            
            # augmentation
            transformed = self.transforms(image = img_org, bboxes=bboxes_org,
                                          bboxes_labels = bboxes_labels_org,
                                          keypoints= keypoints_org_flattened)
          
            img = transformed['image']
            bboxes = transformed['bboxes']

            ### ????????????? 왜 transform이후 keypoints가 그냥 list에 숫자가 담겨진건지



            # transformed["keypoints"] : [1019, 487, 1432, 404, 861, 534, 392, 666]
            # keypoints_transformed_unflattened : [[[1019, 487], [1432, 404]], [[861, 534], [392, 666]]
            keypoints_transformed_flattened = np.reshape(np.array(transformed['keypoints']),
                                                         (-1,2,2)).tolist()
            
            keypoints = []
            for idx, obj in enumerate(keypoints_transformed_flattened):
                obj_keypoints = []
                for k_idx,kp in enumerate(obj):
                    obj_keypoints.append(kp + [keypoints_org[idx][k_idx][2]])

                keypoints.append(obj_keypoints)
                ### 이렇게 되면 keypoints의 형태는 keypoints_org와 똑같은가???????

        else:
            img, bboxes, keypoints = img_org, bboxes_org, keypoints_org


        # transform 통과한 값 -> tensor
        # as_tensor(list->tensor) : 속도이점
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # keypoint label dictonary
        target = {}
        target['boxes'] = bboxes
        target['labels'] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)
        target['img_id'] = torch.tensor([index])
        target['area'] = (bboxes[:,3] - bboxes[:,1]) * (bboxes[:,2] - bboxes[:,2])
        target['iscrowd'] = torch.zeros(len(bboxes), dtype=torch.int64)     #이미지에서 가려진 요소 유뮤
        target['keypoints'] = torch.as_tensor(keypoints, dtype=torch.float32)

        # 이미지도 tensor로 변환
        img = F.to_tensor(img)

        bboxes_org = torch.as_tensor(bboxes_org, dtype=torch.float32)
        target_org = {}
        target_org["boxes"] = bboxes_org
        target_org["labels"] = torch.as_tensor([1 for _ in bboxes_org],
                                                    dtype=torch.int64)  # all objects are glue tubes
        target_org["image_id"] = torch.tensor([index])
        target_org["area"] = (bboxes_org[:, 3] - bboxes_org[:, 1]) * (
                    bboxes_org[:, 2] - bboxes_org[:, 0])
        target_org["iscrowd"] = torch.zeros(len(bboxes_org), dtype=torch.int64)
        target_org["keypoints"] = torch.as_tensor(keypoints_org, dtype=torch.float32)
        img_org = F.to_tensor(img_org)


        if self.demo:
            return img, target, img_org, target_org
        else:
            return img, target
        

    def __len__(self):
        return len(self.img_files)


if __name__ =='__main__':
    test = keypointDataset('computervision/keypointRCNN/keypoint_dataset/train')
    for i in test:
        print(i)