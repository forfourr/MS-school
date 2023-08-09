import json
import os
import shutil
import cv2

# create data folder
os.makedirs('computervision/data/glass_dataset/train/images',exist_ok=True)
os.makedirs('computervision/data/glass_dataset/valid/images',exist_ok=True)
os.makedirs('computervision/data/glass_dataset/train/labels',exist_ok=True)
os.makedirs('computervision/data/glass_dataset/valid/labels',exist_ok=True)

# img path
img_path = 'computervision/data/glass_dataset/valid'

#coco path
coco_anno_path = 'computervision/data/glass_dataset/valid/_annotations.coco.json'

# yolo format ann save path
yolo_anno_path = 'computervision/data/glass_dataset/valid/labels'

# yolo format img save
yolo_img_folder = 'computervision/data/glass_dataset/valid/images'

# coco class name
coco_classes = ['glass']
yolo_classes = {'glass':0}


# loasd COCO anno
with open(coco_anno_path,'r') as f:
    coco_annos = json.load(f)

img_infos = coco_annos['images']
ann_infos = coco_annos['annotations']

for image_info in img_infos :
    image_file_name = image_info['file_name']
    file_name = image_file_name.replace(".jpg", "")
    id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']
    for ann_info in ann_infos :
        if ann_info['image_id'] == id :
# coco -> two stage 1
            # yolo -> one stage 0
            category_id = ann_info['category_id'] -1
            x, y, w, h = ann_info['bbox']

# xywh -> center x center y w h
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w /= image_width
            h /= image_height

# image copy to dst folder
            # Copy image to YOLO image folder
            source_image_path = os.path.join(img_path, image_file_name)
            destination_image_path = os.path.join(yolo_img_folder, image_file_name)
            shutil.copy(source_image_path, destination_image_path)

# write to text file
            yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
            text_path = os.path.join(yolo_anno_path, f"{file_name}.txt")
            with open(text_path, 'a') as f :
                f.write(yolo_line)