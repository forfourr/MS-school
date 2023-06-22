import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
"""
csv파일을 읽고 dataload에 올리는 test
"""

class HeightWeightDataset(Dataset):
    def __init__(self, csv_path):   #class가 실행되는 즉시 실행됨
        self.data =[]
        with open(csv_path,'r',encoding='utf-8') as f:
            next(f)     #첫번째는 헤더이므로 제외
            for line in f:
                print(line)
                _,height,weight = line.strip().split(',')
                height = float(height)
                #weight =float(weight)
                convert_to_kg = round(self.convert_to_kg(weight),2)  #다른 함수 호출/두번째자리까지 반올림
                convert_to_cm = round(self.convert_to_cm(height),1)
                self.data.append([convert_to_cm, convert_to_kg])    #빈 리스트에 추가
    
    def __getitem__(self, index):
        #tensor로 변환
        data = torch.tensor(self.data[index], dtype=torch.float)
        return data
    
    def __len__(self):
        return len(self.data)

    #코드 가독성과/유지보수성을 위해 추가한 함수
    def convert_to_kg(self,weight_lb):
        return float(weight_lb) * 0.453592  #여기서 대신 float해줌
    
    def convert_to_cm(self,height_in):
        return height_in *0.24


if __name__ =='__main__':
    dataset = HeightWeightDataset('data/hw_200.csv')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        x = batch[:,0].unsqueeze(1)   #unsqueeze:1인차원 생성
        y = batch[:,1].unsqueeze(1)
        #print(x,y)

    #print(dataset.data)