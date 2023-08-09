import json
import os
import shutil
import cv2

def to_yolo(mode):
    # save folder
    yolo_anno_path = f'yolov8/ultralytics-main/ultralytics/cfg/datasets/swim_dataset/{mode}/labels'
    yolo_img_path = f'yolov8/ultralytics-main/ultralytics/cfg/datasets/swim_dataset/{mode}/images'

    # create data folder
    os.makedirs(f'yolov8/ultralytics-main/ultralytics/cfg/datasets/swim_dataset/{mode}/images',exist_ok=True)
    os.makedirs(f'yolov8/ultralytics-main/ultralytics/cfg/datasets/swim_dataset/{mode}/labels',exist_ok=True)

    # img path
    img_path = f'data/swim_dataset/{mode}'

    #coco path
    coco_anno_path = f'data/swim_dataset/{mode}/_annotations.coco.json'


    # coco class name
    coco_classes = {
        0: 'aerial-pool',
        1: 'black-hat',
        2: 'body',
        3: 'bodysurface',
        4: 'bodyunder',
        5: 'swimmer',
        6: 'umpire',
        7: 'white-hat'
    }

    # YOLO 클래스 이름 및 클래스 ID 매핑
    yolo_classes = {
        'aerial-pool': 0,
        'swimmer': 1  # Swimmmers are represented by class ID 1 in YOLO format
    }

    # JSON 파일 불러오기
    with open(coco_anno_path, 'r') as f:
        coco_data = json.load(f)

    # YOLO 형식으로 변환하여 저장
    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        image_path = os.path.join(yolo_img_path, file_name)
        img_path_org = os.path.join(img_path,file_name)
        shutil.copy(img_path_org, image_path)

        # 해당 이미지의 어노테이션 정보 추출
        image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        labels_path = os.path.join(yolo_anno_path, file_name.replace('.jpg', '.txt'))
        # YOLO 형식으로 레이블 작성 및 저장
        with open(labels_path, 'w') as label_file:
            for annotation in image_annotations:
                category_id = annotation['category_id']
                bbox = annotation['bbox']

                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                bbox_width = bbox[2] / width
                bbox_height = bbox[3] / height

                yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                
                with open(labels_path, 'a') as f :
                    f.write(yolo_line)
    print("Conversion completed.")

if __name__ =='__main__':
    to_yolo('valid')