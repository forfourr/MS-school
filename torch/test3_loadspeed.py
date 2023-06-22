from typing import Any
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#판독함수
def is_grayscale(img: Image):
    return img.mode=='L'

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform = None):
        self.image_paths = glob.glob(os.path.join(image_paths,'*','*','*.jpg'))     #*jpg인 파일만 받는다
        self.transform = transform
        self.label_dict = {'dew':0, 'fogsmog':1, 'frost':2, 'glaze':3, 'hail':4,'lightning':5,
                           'rain':6, 'rainbow':7, 'rime':8, 'sandstorm':9, 'snow':10}
        self.cache = {}     #이미지 캐시


    def __getitem__(self, index):
        #캐시 메모리 안에 있다면 일일히 로드할 필요 없음
        if index in self.cache:         
            image, label = self.cache[index]
        else:
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert('RGB')   #디스크에서 일일히 로드
            
            if not is_grayscale(image):     #이미지가 흑백아니면 실행
                folder_name = image_path.split('\\')     #tain 폴더에 라벨링 되어 저장되어 있기 떄문에
                #['C:/Users/iiile/Vscode_jupyter/data/sample_data', 'train', 'dew', '2208.jpg']
                folder_name = folder_name[2]    #label만
                label = self.label_dict[folder_name]
                #캐시메모리 올리기
                self.cache[index] = (image, label)
   
            else:
                print(f"{image_path} 파일은 흑백입니다.")
                return None, None   #흑백일 때 오류이므로 none return
        
        if self.transform:      #self트렌스 폼에 무언가 있으면 반환해라
            image = self.transform(image)
        
        return image, label
        
    
    def __len__(self):
        return len(self.image_paths)
    


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])



    image_paths = 'C:/Users/iiile/Vscode_jupyter/data/sample_data/'
    dataset = CustomDataset(image_paths,transform)
    for i in dataset:
        print(i)

    dataLoader = DataLoader(dataset, 32, shuffle=True)       #배치사이즈 메모리에 얼마나 올릴지 -> 메모리 문제 최소화

