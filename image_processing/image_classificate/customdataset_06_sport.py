import os
import cv2
import glob
import pandas as pd

from torch.utils.data import Dataset

class Customdataset(Dataset):
    def __init__(self,data_dir, mode='train', transform=None):
        self.data_dir = glob.glob(os.path.join(data_dir,"*","*.jpg"))
        self.transform = transform

        # csv
        #self.construct_dataset_by_csv(f'{PATH}/sport_dataset/sports.csv',mode)

        # 기존 폴더
        self.label_dict = self.create_label_dict()



        # ## label dict 만드는 방법 1
        # self.label_dict = {}
        # folder_name_list = glob.glob(os.path.join(data_dir,'*'))
        # for i, folder_name in enumerate(folder_name_list):
        #     folder_name = os.path.basename(folder_name)
        #     self.label_dict[folder_name] = i



    ## label dict 만드는 방법 2
    def create_label_dict(self):
        label_dict = {}
        for filepath in self.data_dir:
            # print(filepath)                                          #./train/welding_line/파일명.png
            # print(os.path.dirname(filepath))                         #./train/welding_line  
            label = os.path.basename(os.path.dirname(filepath))        #파일을 포함한 디렉토리만
            
            # label을 dict에 하나씩 추가하면서 len()으로 번호 메김!!!!!
            if label not in label_dict:
                label_dict[label] = len(label_dict)

        return label_dict



    # ### read csv file
    # def construct_dataset_by_csv(self, csv_dir,mode,transformm=None):
    #     # 파일 읽기
    #     self.csv_data = pd.read_csv(csv_dir)
    #     # mode - train/test/val
    #     self.csv_data = self.csv_data.loc[self.csv_data['data set'] == mode]

    #     file_list_csv = self.csv_data['filepaths'].to_list()

    #     # .lnk 확장자 제거
    #     for item in file_list_csv:
    #         if item.endswith(".lnk"):
    #             print("file .ink")
    #             self.csv_data.drop(index=1, axis=0, inplace=True)
    #             self.csv_data.reset_index(drop=True,inplace=True)
        
    #     # label 추출
    #     self.label_col = self.csv_data['labels'].to_list()
    #     self.label_id = self.csv_data['class id'].to_list()


    #    self.transform = transformm
    

    def __getitem__(self, index):
        img_path = self.data_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## 폴더로 사용할 경우
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.label_dict[label_name]

        # ## csv로 사용할 경우
        # label_idx = self.label_id[index]
        # label = self.label_col[index]
        
        #augmentation
        if self.transform is not None:
            img = self.transform(image = img)['image']
        return img, label, img_path

    def __len__(self):
        return len(self.data_dir)
    
    
PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
#PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'

# if __name__ == '__main__':
    
#     test = Customdataset(f"{PATH}/sport_dataset/train",'valid')
#     label_dict = test.create_label_dict()
#     print(label_dict)