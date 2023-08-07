import torch
import torchvision
import numpy as np
import cv2
from torchvision.models.detection.rpn import AnchorGenerator
from customdataset import keypointDataset
from utils import collate_fn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings(action='ignore')


def get_mode(num_keypoints, weight_path = None):
    anchor_generate = anchor_generate(size=(32, 64, 128, 256, 512),
                                      aspect_rations = (0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints= num_keypoints,
                                                                   num_classes = 2,
                                                                   rpn_anchor_generator= anchor_generate)
    
    if weight_path:
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict)

    return model


def visualize(image, bboxes, keypoints):

    keypoints_classes = {0: 'Head', 1: 'Tail'}

    # resize(800*800) -> sacle
    height, width = image.size[:2]
    scale_factor = min(800/width, 800/height)

    new_width, new_height = int(width* scale_factor), int(height*scale_factor)
    image_resize = cv2.resize(image, (new_width, new_height))

    # bbox scale
    bbox_sclae = [[int(coord * scale_factor)
                   for coord in bbox] for bbox in bboxes]
    keypoints_scale = [[[int(coord[0]* scale_factor), int(coord[1]*scale_factor)]
                        for coord in kps]
                       for kps in keypoints]
    
    for bbox in bbox_sclae:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])

        image_resize = cv2.rectangle(image, start_point, end_point,(255,0,0),2)

    ## end_point(kps)??????????출력
    for kps in keypoints_scale:
        for idx, kp in enumerate(kps):
            image_resize = cv2.circle(image_resize,tuple(kp), 5, (255,0,0),5)
            image_resize = cv2.putText(image_resize.copy()," "+ keypoints_classes[idx],
                                       tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2, cv2.LINE_AA)
            

    cv2.imshow('test', image_resize)
    cv2.waitKey(0)


def calculate_iou(box1, box2):
    # 겹치는 영역 계산
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])
    inter_area = max(0, xmax_inter- xmin_inter) * max(0, ymax_inter-ymin_inter)

    area_box1 = (box1[2]- box1[0])*(box1[3]- box1[1])
    area_box2 = (box2[2]- box2[0])*(box2[3]- box2[1])

    union_area = area_box1+ area_box2 - inter_area

    iou = inter_area/union_area
    return iou

# 중복된 박스 제거
def filter_duplicated_boxes(boxes, iou_threshold=0.5):
    filtered_boxes = []

    for i, box1 in enumerate(boxes):
        is_duplicate = False
        for j, box2 in enumerate(filtered_boxes):
            print(f"IoU: {calculate_iou(box1, box2)}")
            if calculate_iou(box1, box2) > iou_threshold:
                # 겹침
                is_duplicate = True
                break
        # 안겹치면
        if not is_duplicate:
            filtered_boxes.append(box1)
    return filtered_boxes


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    KEYPOINT_FOLDER = 'computervision/keypointRCNN/'
    dataset_test = keypointDataset(KEYPOINT_FOLDER,)