import torch
import torch.nn as nn
import torchvision
from torchvision.models.mobilenetv2 import mobilenet_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset_04_food import Customdataset
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os


PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'


'''['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani',
'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'ku
lfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', '
samosa']'''


def train(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    


    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch}", leave=False)
        
        ################# Train #################
        print("Start Train")
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            ## train acc
            _, pred = torch.max(output, 1)
            train_acc += (pred == target).sum().item()

            train_loader_iter.set_postfix(f"Loss: {loss.item()}")

        train_losses.append(train_losses/len(train_loader))
        train_accs.append(train_acc/len(train_loader.dataset))


        ################# Test #################
        print('Start test')
        model.eval()

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.float().to(device), target.to(device)

                output = model(data)
                _, pred = torch.max(output,1)

                val_acc += (pred == target).sum().item()
                val_loss += criterion(output, target).item()

        val_losses.append(val_loss/len(val_loader))
        val_accs.append(val_acc/len(val_loader.datatset))
        
        ################# Save model #################
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), f"{PATH}/ex04_food_mobilenet_v2.pt")
            best_val_acc = val_acc

        print(f"Epoch [{epoch + 1} / {epochs}] , Train loss [{train_loss:.4f}],"
              f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
              f"Val ACC [{val_acc:.4f}]")
        

    torch.save(model.state_dict(), f"{PATH}/ex04_food_mobilenet_v2.pt")

    ################# Save model #################
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()



    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilenet_v2(pretrained=True)
    # (1): Linear(in_features=1280, out_features=1000, bias=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 20)
    model.to(device)


    # Augmentation
    train_transforms = A.transforms.Compose([
        A.SmallestMaxSize(max_size=220),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6),
        A.RandomShadow(),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])
    val_transforms = A.transforms.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    train_dataset = Customdataset(f'{PATH}/food_dataset/train',
                                transfrom=train_transforms)
    val_dataset = Customdataset(f"{PATH}/food_dataset/validation",
                                transfrom=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=False)

    #loss, optim
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    train(model, train_loader, val_loader, epochs, optimizer, criterion, device)
        






if __name__ == '__main__':
    
    main()