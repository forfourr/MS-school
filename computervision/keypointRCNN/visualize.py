import cv2
import albumentations as A
from customdataset import keypointDataset
from torch.utils.data import DataLoader
from utils import collate_fn
import matplotlib.pyplot as plt
import numpy as np


train_transform = A.Compose([
    A.Sequential([
        A.RandomRotate90(p=1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3,
                                brightness_by_max=True, always_apply=False, p=1)
    ], p=1),
], keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']))




dataset= keypointDataset('computervision/keypointRCNN/keypoint_dataset/train',
                         transforms=train_transform,demo=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

keypoints_classes = {0:'Head', 1:'Tail'}

def visualize(img, bboxes, keypoints, img_org=None, bboxes_org=None, keypoints_org=None):
    font_size=18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point  = (bbox[2], bbox[3])

        img = cv2.rectangle(img.copy(), start_point, end_point, (255,0,0,),2 )

    for kpts in keypoints:
        for idx, kp in enumerate(kpts):
            img = cv2.circle(img.copy(), tuple(kp), 5, (255,0,0),10)
            img = cv2.putText(img.copy(), f"{keypoints_classes[idx]}",
                                tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3, cv2.LINE_AA)
            
    if img_org is None and keypoints_org is None:
        plt.figure(figsize=(40,40))
        plt.imshow(img)

    else:
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point  = (bbox[2], bbox[3])

            img = cv2.rectangle(img.copy(), start_point, end_point, (255,0,0,),2 )

        for kpts in keypoints:
            for idx, kp in enumerate(kpts):
                img = cv2.circle(img, tuple(kp), 5, (255,0,0),10)
                img = cv2.putText(img, f"{keypoints_classes[idx]}",
                                    tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3, cv2.LINE_AA)

        f, ax = plt.subplots(1,2, figsize=(40,20))

        ax[0].imshow(img_org)
        ax[0].set_title('org', font_size)
        ax[1].imshow(img)
        ax[1].set_title('trnasformed', font_size)
        plt.show()


if __name__ == '__main__':
    
    visualize_image_show = True
    visualize_targets_show = True

    image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # CustomDataset에서 Tensor로 변환했기 때문에 다시 plt에 사용할 수 있도록 numpy 행렬로 변경
    # img, target, img_org, target_org = batch이므로, batch[0]는 img를 지칭
    # batch[0][0]에 실제 이미지 행렬에 해당하는 텐서가 있을것 (batch[0][1]에는 dtype 등의 다른 정보가 있음)
    bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
    # target['boxes']에 bbox 정보가 저장되어있으므로, 해당 키로 접근하여 bbox 정보를 획득

    keypoints = []
    for kpts in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints.append([kp[:2] for kp in kpts])
        # 이미지 평면상 점들이 필요하므로, 3번째 요소로 들어있을 1을 제거

    image_org = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # batch[2] : image_org
    bboxes_org = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
    # batch[3] : target

    keypoints_org = []
    for kpts in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints_org.append([kp[:2] for kp in kpts])

    if visualize_image_show:
        visualize(image, bboxes, keypoints, image_org, bboxes_org, keypoints_org)
    if visualize_targets_show and visualize_image_show == False:
        print("Org targets: \n", batch[3], "\n\n")
        # org targets: (줄바꿈) org targets dict 출력 (두줄 내림)
        print("Transformed targets: \n", batch[1])