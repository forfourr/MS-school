import json
import os
import cv2
import shutil
import random
from PIL import Image
from sklearn.model_selection import train_test_split

#json경로 가져오기
PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
org_folder = f"{PATH}/HW_dataset/images"
json_path = f'{PATH}/HW_data/anno/annotation.json'


with open(json_path, 'r', encoding='utf-8') as j:
    json_data = json.load(j)

train_path = os.path.join(PATH,'HW_data','train')
val_path = os.path.join(PATH,'HW_data','val')

    

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


    # 이미지 crop
    img_path = os.path.join(org_folder,filename)
    img = Image.oepn(img_path)
    img = img.conver('RG')

    cropped_image = img.crop((x1,y1,x2,y2))
    
    
    # image padding
    width_, height_ = cropped_image.size
    if


    # image save
    # 랜덤하게 저장
    if np.random.rand() < train_ratio:



    # 이미지 이동
    








# # Mel:2997, STFT:2997, waveshow:2997
# train_data, val_data = train_test_split(file_path_01, test_size=0.2)
# stft_train_data, stft_val_data = train_test_split(file_path_02, test_size=0.2)
# mel_train_data, mel_val_data = train_test_split(file_path_03, test_size=0.2)


# move_temp = ImageDateMove(json_path, train_dir, val_dir,test_dir)
# move_temp.move_images()
