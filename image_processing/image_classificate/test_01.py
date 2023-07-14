import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
# 본인이 학습한 모델과 맞는것으로 선택
# 선생님이 주신 모델은 resnet으로 학습시켰기 때문에 resnet사용
from torchvision.models import resnet18
from customdataset_01 import CustomDataset

import cv2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
    
    #### 모델 불러오기
    model = resnet18()
    # feature 맞추기
    # (fc): Linear(in_features=512, out_features=1000, bias=True)
    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 3)

    #### .pt file load 학습시킨 모델 불러오기
    model.load_state_dict(torch.load(f=f'{PATH}/sound_model.pt'))

    #### transfrom
    # 학습했을 때와 같은 transform을 줘야함
    val_transfrom = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset(f"{PATH}/sound/val",
                                    val_transfrom)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)     
    #batch_Szie=n으로 주면 n씩 묶여 나옴

    ### Label dict
    label_dict = {0: 'Mel',
                1:"STFT",
                2:"waveshow"}
    
    
    model.to(device)
    model.eval()
    corr = 0
    from tqdm import tqdm   #진행률 알기위해
    with torch.no_grad():
        for data,target, path in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)            #각 라벨 별 예측값이 있는 list
            _, pred = torch.max(output, 1)  # 그 중에 가장 높은 값을 가진 라벨
            corr += (pred == target).sum().item()

            # # 무슨 파일을 뭐로 예측했는자 확인
            # img = cv2.imread(path[0])           # path가 tuple type이기 때문에 path[0] -> str
            # img = cv2.resize(img,(400,400))
            
            # pred_label = label_dict[pred.item()]
            # target_label = label_dict[target.item()]

            # pred_txt = f"pred: {pred_label}"
            # target_txt = f"target: {target_label}"
            # img = cv2.rectangle(img, (0,0),(400,80),(255,255,255),-1)
            # img = cv2.putText(img, pred_txt, (20,20), cv2.FONT_ITALIC, 1,(255,0,5), 2)
            # img = cv2.putText(img, target_txt, (20,50), cv2.FONT_ITALIC, 1,(0,0,255), 2)

            # cv2.imshow('test',img)
            # if cv2.waitKey() ==ord('q'):
            #     exit()




    corr_per = corr/len(test_loader.dataset)*100
    print(f"Test Accurracy : [{corr}/{len(test_loader.dataset)}], {corr_per:.3f}%")





    pass


if __name__ == '__main__':
    main()