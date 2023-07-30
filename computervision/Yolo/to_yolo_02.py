import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

# folder path
train_foler_path = 'computervision/data/candy_dataset/train'
val_foler_path = 'computervision/data/candy_dataset/val'

# csv path
train_csv_path = os.path.join(train_foler_path, 'annotation.csv')
val_csv_path = os.path.join(val_foler_path,'annotation.csv')

# csv -> df
train_anno_df = pd.read_csv(train_csv_path)
val_anno_df = pd.read_csv(val_csv_path)


# image resize, bounding box scale
def resize_and_scale_bbox(img, bbox, target_size):
    img_width , img_height = img.size

    # resize
    img = img.resize(target_size, Image.LANCZOS)    #고급 리샘플링 알고리즘

    # bbox scale
    x, y, width, height=  bbox
    x_scale = target_size[0]/img_width
    y_scale = target_size[1]/img_height

    x_center = (x+width /2) * x_scale
    y_center = (y+height/2) * y_scale
    scaled_width = width * x_scale
    scaled_height = height * y_scale

    scaled_bbox = (x_center, y_center, scaled_width, scaled_height)
    return img, scaled_bbox

# turn to YOLO format
def convert_to_yolo_format(anno_df, org_folder, output_folder, target_size):
    for idx, row in tqdm(anno_df.iterrows()):
        img_name = row['filename']
        label = row['region_id']

        img_path = os.path.join(org_folder, img_name)
        new_img_path = os.path.join(output_folder,'images',f"{img_name}")

        shape_attr = json.loads(row['region_shape_attributes'])
        x = shape_attr['x']
        y = shape_attr['y']
        width = shape_attr['width']
        height = shape_attr['height']

        img = Image.open(img_path)

        img, scaled_bbox = resize_and_scale_bbox(img, (x,y,width,height), target_size)

        img.save(new_img_path)

        x_center, y_center, width, height = scaled_bbox
        x_center /= target_size[0]
        y_center /= target_size[1]
        norm_width = width / target_size[0]
        norm_height = height / target_size[1]

        class_id = label

        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_folder, 'labels',label_file)

        with open(label_path, 'a') as f:
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
            f.write(line)


## 폴더 생성
train_yolo_folder = 'computervision/data/candy_dataset/yolo_dataset/train'
val_yolo_folder = 'computervision/data/candy_dataset/yolo_dataset/val'
os.makedirs(os.path.join(train_yolo_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_yolo_folder,'labels'), exist_ok=True)
os.makedirs(os.path.join(val_yolo_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_yolo_folder,'labels'), exist_ok=True)


# target size
target_size = (1280,720)

# yolo 변환
convert_to_yolo_format(train_anno_df, train_foler_path, train_yolo_folder, target_size)
convert_to_yolo_format(val_anno_df, val_foler_path, val_yolo_folder, target_size)