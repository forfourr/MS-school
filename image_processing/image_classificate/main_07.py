##### class arg 사용

import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.resnet import resnet50
from torchvision.models.efficientnet import efficientnet_b0
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset_06_sport import Customdataset
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


    def train(self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0):
        best_acc = 0.0
        
        ### train
        print("Start training")
        for epoch in range(start_epoch, epochs):
            train_acc =0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch: {epoch+1}/{epochs}"), leave=False)
            
            for i, (data, label) in enumerate(train_loader_iter):
                data, label = data.float().to(self.device), label.to(self.device)

                optimizer.zero_grad()

                output = self.model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _,pred = torch.max(output, 1)
                train_acc += (pred == label).sum().item()

            self.train_losses.append(train_loss/len(train_loader))
            self.train_accs.append(train_acc/len(train_loader.datast))

            print(len(train_loader), len(train_loader.dataset))

            ### eval
            print("start eval")
            self.model.eval()
            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.float().to(self.device), label.to(self.device)

                    output = self.model(data)

                    loss = criterion(output, label)
                    _,pred = torch.max(output, 1)

                    val_loss += loss.item
                    val_acc += (pred == label).sum().item()

            val_acc /= len(val_loader.dataset)
            self.val_losses.append(val_loss/len(val_loader))
            self.val_accs.append(val_acc)

            print(f"Epoch [{epoch + 1} / {epochs}] , Train loss [{train_loss:.4f}],"
                    f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
                    f"Val ACC [{val_acc:.4f}]")

            ### best acc 
            if val_acc > best_acc:
                torch.save(self.model.state_dict(),
                           f"{PATH}/07_license_efficient.pt")
                best_acc = val_acc

            ### save model
            torch.save({
                'epoch':epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accs': self.train_accs,
                'train_losses': self.train_losses,
                'val_accs': self.val_accs,
                'val_losses': self.val_losses
            }, f"{PATH}/weight/07_license_efficient_checkpoint.pt")

        torch.save(self.model.state_dict(), f"{PATH}/07_license_efficient.pt")


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

    def run(self, args):
        self.model = efficientnet_b0(pretrained=True)
        #Linear(in_features=1280, out_features=1000, bias=True)
        in_features = self.model.classifier[1]._in_features
        self.model.classifier[1] = nn.Linear(in_features, 50)

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--train_dir", type=str, default=f"{PATH}/license_dataset/train")
    parser.add_argument("--val_dir", type=str, default=f"{PATH}/license_dataset/valid")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default=f"{PATH}/weight")

    # parameter
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-2)


    parser.add_argument("--resume_training", action='store_true')   #명령을 주면 true,아니면 false
    parser.add_argument("--checkpoint_path", type=str,
                        default=f"{PATH}/weight/07_efficient_checkpoint.pt")



    args = parser.parse_args()
        
    weight_folder = args.checkpoint_folder_path
    os.makedirs(weight_folder,exist_ok=True)

    train_dir = args.train_dir
    val_dir = args.val_dir

    #classifier = Classifier()
    model = efficientnet_b0(pretrained=True)
    print(model.classifier[1])
        



