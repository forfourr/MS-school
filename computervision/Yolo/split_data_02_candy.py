import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def split(anno_df, path, names):
    split_anno = pd.DataFrame(columns=anno_df.columns)

    for img_name in names:
        # image 이동
        img_path = os.path.join(img_folder_path,img_name)       # 원래 이미지 위치
        new_img_path = os.path.join(path, img_name)  # 이동 할 이미지 위치

        shutil.copy(img_path, new_img_path)

        # 바운딩 박스 정보 복사
        anno = anno_df.loc[anno_df['filename'] == img_name].copy()
        anno['filename'] = img_name
        split_anno = split_anno._append(anno)

    # csv 파일로 저장
    split_anno.to_csv(os.path.join(path, 'annotation.csv'), index=False)



if __name__ == "__main__":
        
    # path
    img_folder_path = 'computervision/data/candy_dataset/images'
    anno_folder_path = 'computervision/data/candy_dataset/annotations'

    csv_path = os.path.join(anno_folder_path,'annotations.csv')


    # create new folder
    train_folder = 'computervision/data/candy_dataset/train'
    val_folder = 'computervision/data/candy_dataset/val'
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    #csv -> df
    anno_df = pd.read_csv(csv_path)

    image_names = anno_df['filename'].unique()    # 중복되지 않는 고유 값
    train_name, eval_name = train_test_split(image_names, test_size=0.2)

    '''
    image name len 528
    train size 422
    eval size 106
    '''

    # train
    split(anno_df, path=train_folder, names=train_name)
    # val
    split(anno_df, path=val_folder, names=eval_name)