from sklearn.model_selection import train_test_split
import glob
import os
import cv2
import shutil
import random

PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
label_folder_path = f"{PATH}/Plant_dataset"

# train/val folder
train_folder_path = os.path.join(PATH,'new_plant','train')
val_folder_path = os.path.join(PATH,'new_plant','val')

os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)


org_folders = os.listdir(label_folder_path)
'''
['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge', '
PalmerAmaranth', 'Prickly Sida', 'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge
', 'SpurredAnoda', 'Swinecress', 'Waterhemp']
'''
for org_folder in org_folders:
    org_folder_full_path = os.path.join(label_folder_path,org_folder)
    # image random shuffle
    images = os.listdir(org_folder_full_path)
    random.shuffle(images)

    # create label folder in tran/val folder
    train_label_folder_path = os.path.join(train_folder_path,org_folder)
    val_label_folder_path = os.path.join(val_folder_path, org_folder)
    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)

    # move dataset -> tran
    split_index = int(len(images)*0.9)
    for image in images[:split_index]:
        src_path = os.path.join(org_folder_full_path, image)       # origin path
        dst_pth = os.path.join(train_label_folder_path, image)
        shutil.copyfile(src_path,dst_pth)


    # move dataset -> val
    for image in images[split_index:]:
        src_path = os.path.join(org_folder_full_path, image)       # origin path
        dst_pth = os.path.join(val_label_folder_path, image)
        shutil.copyfile(src_path,dst_pth)

print('COMPLETE')