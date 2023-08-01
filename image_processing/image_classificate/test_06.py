## food dataset

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from customdataset_06_sport import Customdataset
import cv2
import glob
import os


def main():
    PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    
    ## label dict 만드는 방법 
    data_dir = f"{PATH}/sport_dataset/valid"
    label_dict = {}
    folder_name_list = glob.glob(os.path.join(data_dir,'*'))
    for i, folder_name in enumerate(folder_name_list):
        folder_name = os.path.basename(folder_name)
        label_dict[i] = folder_name


    ### model
    
    model = resnet50(pretrained=True)
    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 100)
    
    ### import trained model
    checkpoint = torch.load(f=f"{PATH}/weight/sport_classifier_checkpoint_19.pt")
    model_state_dict = checkpoint['model_state_dict']

    # print(list(model.parameters()))

    ### transfrom
    val_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])


    
   
    val_dataset = Customdataset(f"{PATH}/sport_dataset/valid",
                                transform = val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



    ("START")
    ### val
    model.to(device)
    model.eval()
    corr = 0
    with torch.no_grad():
        for data, target, path in val_loader:
            data, target = data.float().to(device), target.to(device)
            target_ = target.item()
            output = model(data)

            _, pred = torch.max(output, 1)
            corr += (pred == target).sum().item()
            
            # show incorrect img
            # if pred != target:
            #     target_label = label_dict[target_]
            #     pred_label = label_dict[pred.item()]
            #     true_label = f"True: {target_label}"
            #     pred_label = f"Pred: {pred_label}"

            #     img = cv2.imread(path[0])
            #     img = cv2.resize(img, (500,500))
            #     img = cv2.rectangle(img, (0,0), (500,100), (255,255,255),-1)
            #     img = cv2.putText(img, pred_label, (0,30), cv2.FONT_ITALIC,1,(255,0,0),2)
            #     img = cv2.putText(img, true_label, (0,75), cv2.FONT_ITALIC,1,(255,0,0),2)

            #     cv2.imshow('test', img)
            #     if cv2.waitKey() == ord('q'):
            #         exit()
            
    corr_per = corr/len(val_loader.dataset)*100
    print(f"Test Accurracy : [{corr}/{len(val_loader.dataset)}], {corr_per:.3f}%")





if __name__ == '__main__':
    main()
