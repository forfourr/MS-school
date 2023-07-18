import json
import os
import cv2
import shutil
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as F

#json경로 가져오기
#PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'
org_folder = f"{PATH}/HW_dataset/images"
json_path = f'{PATH}/HW_dataset/anno/annotation.json'


with open(json_path, 'r', encoding='utf-8') as j:
    json_data = json.load(j)

train_path = os.path.join(PATH,'HW_data','train')
val_path = os.path.join(PATH,'HW_data','val')
test_path = os.path.join(PATH,'HW_data','test')





for key, value in json_data.items():
    filename = value['filename']
    width = value['width']
    height = value['height']
    bbox = value['anno'][0]['bbox']     #[1738, 806, 1948, 993]
    x1, y1, x2, y2 = bbox
    

    # Label별 폴더 생성
    label = value['anno'][0]['label']

    train_label_dir = os.path.join(train_path,label)
    os.makedirs(train_label_dir, exist_ok=True)
    val_label_dir = os.path.join(val_path, label)
    os.makedirs(val_label_dir, exist_ok=True)
    test_label_dir = os.path.join(test_path, label)
    os.makedirs(test_label_dir, exist_ok=True)


    # 이미지 crop
    img_path = os.path.join(org_folder,filename)
    img = Image.open(img_path)
    img = img.convert('RGB')

    cropped_image = img.crop((x1,y1,x2,y2))
    
    
    # image padding
    width_, height_ = cropped_image.size
    if width_ > height_:
        padding_img = Image.new(cropped_image.mode, (width_,width_),(0,))
        padding = (0, int((width_ - height_)/2))    #좌우, 상하
    else:
        padding_img = Image.new(cropped_image.mode, (height_,height_), (0,))
        padding = (int((height_ - width_)/2), 0)    #상하, 좌우
    
    padding_img.paste(cropped_image, padding)
    # padding_img.show() -> 이미지 확인


    # image resize
    resized_img = F.resize(padding_img, (225,225))



    # image save
    # 랜덤하게 저장
    train_ratio = 0.8
    val_ratio = 0.9
    if np.random.rand() < train_ratio:
        print(np.random.rand())
        save_dir = os.path.join(train_path, label)
    elif train_ratio <= np.random.rand()< val_ratio:
        print(np.random.rand())
        save_dir = os.path.join(val_path, label)
    else:
        print(np.random.rand())
        save_dir = os.path.join(test_path, label)
        
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}_{label}_{bbox}.png")
    resized_img.save(save_path)


