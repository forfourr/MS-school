## food dataset

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from customdataset_03_plant import CustomDataset
import cv2

def main():
    PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    label_dict = {0: 'Carpetweeds ', 1: "Crabgrass", 2: 'Eclipta', 3: 'Goosegrass',
                    4: 'Morningglory', 5: 'Nutsedge', 6 : 'PalmerAmaranth', 7: 'Prickly Sida',
                    8: 'Purslane', 9: 'Ragweed', 10: 'Sicklepod', 11: 'SpottedSpurge',
                    12: 'SpurredAnoda', 13: 'Swinecress', 14: 'Waterhemp'
    }


    ### model
    model = mobilenet_v2()
    # (1): Linear(in_features=1280, out_features=1000, bias=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 15)
    
    ### import trained model
    model.load_state_dict(torch.load(f=f"{PATH}/ex01_0714_best_mobilenet_v2.pt"))
    # print(list(model.parameters()))

    ### transfrom
    val_transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
   
    val_dataset = CustomDataset(f"{PATH}/new_plant/val",
                                transform = val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



    ("START")
    ### val
    model.to(device)
    model.eval()
    corr = 0
    with torch.no_grad():
        for data, target, path in val_loader:
            data, target = data.to(device), target.to(device)
            target_ = target.item()
            output = model(data)

            _, pred = torch.max(output, 1)
            corr += (pred == target).sum().item()
            
            # show incorrect img
            if pred != target:
                target_label = label_dict[target_]
                pred_label = label_dict[pred.item()]
                true_label = f"True: {target_label}"
                pred_label = f"Pred: {pred_label}"

                img = cv2.imread(path[0])
                img = cv2.resize(img, (500,500))
                img = cv2.rectangle(img, (0,0), (500,100), (255,255,255),-1)
                img = cv2.putText(img, pred_label, (0,30), cv2.FONT_ITALIC,1,(255,0,0),2)
                img = cv2.putText(img, true_label, (0,75), cv2.FONT_ITALIC,1,(255,0,0),2)

                cv2.imshow('test', img)
                if cv2.waitKey() == ord('q'):
                    exit()
            
    corr_per = corr/len(val_loader.dataset)*100
    print(f"Test Accurracy : [{corr}/{len(val_loader.dataset)}], {corr_per:.3f}%")





if __name__ == '__main__':
    main()
