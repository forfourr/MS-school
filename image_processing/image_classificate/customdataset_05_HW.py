import os
import cv2
import glob

from torch.utils.data import Dataset



class Customdataset(Dataset):
    def __init__(self, data_dir, transfrom=None):
        # data_dir = './data/food_dataset/train
        self.data_dir = glob.glob(os.path.join(data_dir,"*","*.png"))
        self.transform = transfrom
        self.label_dict = self.create_label_dict()

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



    def __getitem__(self, index):
        img_path = self.data_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = os.path.basename(os.path.dirname(img_path))
        label_idx = self.label_dict[label]

        #augmentation
        if self.transform is not None:
            img = self.transform(image = img)['image']

        return img, label_idx, img_path

    def __len__(self):
        pass

#PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'

if __name__ == '__main__':
    
    test = Customdataset(f"{PATH}/HW_data/train")
    # for i in test:
    #     pass