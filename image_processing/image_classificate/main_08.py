##### class arg 사용

import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision.models.mobilenetv2 import mobilenet_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset_08_Hw_food import Customdataset
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os
import argparse
PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
#PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'

class Classifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train(self,train_loader, val_loader, epochs, optimizer, criterion, start_epoch):
        
        best_val_acc = 0.0

        for epoch in range(start_epoch, epochs):
            train_loss , train_acc =0.0, 0.0
            val_loss, val_acc = 0.0, 0.0

            ### train
            print("start train")
            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch: {epoch+1}/{epochs}"), leave=False)

            for i, (data, label) in enumerate(train_loader_iter):
                data , label = data.float().to(self.device), label.to(self.device)

                optimizer.zero_grad()

                output = self.model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, pred = torch.max(output, 1)
                train_acc += (pred ==label).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_acc / len(train_loader.dataset)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            ### valid
            print("start valid")
            self.model.eval()
            with torch.no_grad():
                for data, label in val_loader:
                    data , label = data.float().to(self.device), label.to(self.device)

                    output = self.model(data)

                    loss = criterion(output, label)
                    val_loss += loss.item()

                    _,pred = torch.max(output, 1)
                    val_acc += (pred == label).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_acc / len(val_loader.dataset)

            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f"Epoch [{epoch + 1} / {epochs}] , Train loss [{train_loss:.4f}],"
                    f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
                    f"Val ACC [{val_acc:.4f}]")
            

            ### best acc 
            if val_acc > best_val_acc:
                torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict(),
                            # 다음에 checkpoint를 이어서 학습하기 위해 필요한 값들
                            "train_losses": self.train_losses,
                            "train_accs": self.train_accs,
                            "val_losses": self.val_losses,
                            "val_accs": self.val_accs
                        }, args.checkpoint_path.replace(".pt",
                                                        "_best.pt"))
                best_val_acc = val_acc

            ### save model
            torch.save({
                'epoch':epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accs': self.train_accs,
                'train_losses': self.train_losses,
                'val_accs': self.val_accs,
                'val_losses': self.val_losses
            },  args.checkpoint_path.replace('.pt', f"_{epoch}.pt"))

        #torch.save(self.model.state_dict(), f"{PATH}/08_hw_efficient.pt")


        self.save_result_to_csv()   # csv저장 함수
        self.plot_loss()            #시각화 함수
        self.plot_acc()

    def save_result_to_csv(self):
        df = pd.DataFrame({
            'Train loss': self.train_losses,
            'Train acc': self.train_accs,
            'Valid loss': self.val_losses,
            'Valid acc': self.val_accs
        })
        df.to_csv(f"{PATH}/07_train_val.csv", index=False)


    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='val loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'{PATH}/loss_plot.png')

    def plot_acc(self):
        plt.figure()
        plt.plot(self.train_accs, label='train acc')
        plt.plot(self.val_accs, label='val acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig(f'{PATH}/acc_plot.png')




            



    def run(self,args):
         ### 모델 선언
        self.model = mobilenet_v2(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 251)
        self.model.to(self.device)

        ### optim, criterion
        optimizer = AdamW(self.model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        criterion = CrossEntropyLoss().to(self.device)

        ### transfrom
        train_transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6),
            A.RandomShadow(),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            ToTensorV2()
        ])

        train_dataset = Customdataset(args.train_dir, 'train',transform=train_transforms)
        val_dataset = Customdataset(args.val_dir,'valid',transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        epochs = args.epochs
        start_epoch = 0

        # 학습시키다 중간에 다시 시작할 때 일정한 값을 가지고 와야한다.
        if args.resume_training:
            checkpoint = torch.load(args.checkpoint_path)               # checkpoint불러옴
            self.model.load_state_dict(checkpoint['model_state_dict'])      # 모델에 대한 정보
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.val_losses = checkpoint['val_losses']
            self.val_accs = checkpoint['val_accs']
            start_epoch = checkpoint['epoch']       #해당 에포크 지점부터 시작

        self.train(train_loader, val_loader, epochs, optimizer, criterion, start_epoch)






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--train_dir", type=str, default=f"{PATH}/08_food_HW_data/train_new")
    parser.add_argument("--val_dir", type=str, default=f"{PATH}/08_food_HW_data/valid_new")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default=f"{PATH}/weight")

    # parameter
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-2)


    parser.add_argument("--resume_training", action='store_true')   #명령을 주면 true,아니면 false
    parser.add_argument("--checkpoint_path", type=str,
                        default=f"{PATH}/weight/08_HW_efficient_checkpoint.pt")



    args = parser.parse_args()

    weight_folder = args.checkpoint_folder_path
    os.makedirs(weight_folder,exist_ok=True)

    train_dir = args.train_dir
    val_dir = args.val_dir
   

    classifier = Classifier()
    classifier.run(args)
    