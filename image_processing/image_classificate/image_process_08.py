import os
import cv2
import glob
import pandas as pd
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

#PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'



labels_dict={}
labels_dict2 = {}
txt_file_path = f"{PATH}/08_food_HW_data/class_list.txt"

def main(mode):

    ## txt file -> label_dict
    with open(txt_file_path, 'r') as file:
        for idx, line in enumerate(file):
            label = line.strip().split(' ')[1]  #라벨 추출
            labels_dict[label] = idx            # key (라벨 네임), values (값)
            labels_dict2[idx] = label           # key (번호), values (라벨 이름)
            # create label folder
            os.makedirs(f"{PATH}/08_food_HW_data/{mode}_new/{label}", exist_ok=True)

    ## csv file -> filename/label dict
    match_dict = {}
    csv_path = f"{PATH}/08_food_HW_data/{mode}_labels.csv"
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)    #첫번째 헤더 제외
        for row in reader:
            img_name, label = row
            match_dict[img_name] = int(label)
        


    img_path = glob.glob(os.path.join(PATH, '08_food_HW_data',mode, "*.jpg"))   #train/valid
    for path in img_path:
        img_name = os.path.basename(path)
        label_idx = match_dict[img_name]
        label= labels_dict2[label_idx]

        #print(img_name, label, label_idx)

        img = Image.open(path)
        img.convert('RGB')

   
        # image padding
        width_, height_ = img.size
        if width_ > height_:
            padding_img = Image.new(img.mode, (width_,width_),(0,))
            padding = (0, int((width_ - height_)/2))    #좌우, 상하
        else:
            padding_img = Image.new(img.mode, (height_,height_), (0,))
            padding = (int((height_ - width_)/2), 0)    #상하, 좌우
        
        padding_img.paste(img, padding)
        # padding_img.show() -> 이미지 확인


        # image resize
        resized_img = F.resize(padding_img, (225,225))
        #resized_img.show()

        ## save img
        save_path = os.path.join(PATH, "08_food_HW_data", f"{mode}_new",label,img_name)
        resized_img.save(save_path)
        


if __name__ == '__main__':
    main('train')
    #main('valid')