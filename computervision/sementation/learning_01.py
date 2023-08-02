import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os
import argparse


class SegLearner():
    def __init__(self, model, optimzier, criterion):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimzier
        self.criterion = criterion.to(self.device)

        self.start_epochs = 0

        self.train_losses= []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []


    def train(self,train_loader,valid_loader, args):
        best_val_acc = 0.0

        print("start train")
        for epoch in range(self.start_epochs, args.epochs):
            train_acc = 0.0
            train_loss = 0.0
            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch: {epoch+1}/{args.epochs}"), leave=False)

            for i, (images, labels) in enumerate(train_loader_iter):
                images = images.float().to(self.device)
                labels = labels.argmax(dim=-1)
                labels = labels.to(self.device)
                
                outputs = self.model(images)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs['out'], labels)
                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(outputs['out'].data,1)
                train_acc += (pred == labels).sum().item()

                train_loss += loss.item()* images.size(0)

            avg_train_acc = train_acc.double() / len(train_loader.dataset)
            avg_train_loss = train_loss / len(train_loader)

            # Append average accuracy and loss to the respective lists
            self.train_accs.append(avg_train_acc)
            self.train_losses.append(avg_train_loss)
            
            print('start eval')
            if epoch%1 ==0:
                self.model.eval()
                with torch.no_grad():
                    val_acc = 0.0
                    val_loss = 0.0
                    for images,labels in valid_loader:
                        images= images.float().to(self.device) 
                        labels = labels.argmax(dim=-1)
                        labels = labels.to(self.device)


                        outputs = self.model(images)
                        
                        loss = self.criterion(outputs['out'],labels)
                        _, pred = torch.max(outputs['out'].data,1)

                        val_acc += (pred == labels).sum().item()
                        val_loss += loss.item()* images.size(0)

                avg_val_acc = val_acc.double()/len(valid_loader.dataset)
                avg_val_loss = val_loss/len(valid_loader)

                self.val_accs.append(avg_val_acc)
                self.val_losses.append(avg_val_loss)

                ## save model ##
                if val_acc > best_val_acc:
                    torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            # 다음에 checkpoint를 이어서 학습하기 위해 필요한 값들
                            "train_losses": self.train_losses,
                            "train_accs": self.train_accs,
                            "val_losses": self.val_losses,
                            "val_accs": self.val_accs
                        }, args.checkpoint_path.replace(".pt",
                                                        "_best.pt"))
                    best_val_acc = val_acc

                print(f"Epoch [{epoch + 1} / {args.epochs}] , Train loss [{train_loss:.4f}],"
                    f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
                    f"Val ACC [{val_acc:.4f}]")
                
            torch.save({
                "epoch": epoch,
                "model_state_dict":self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "train_accs": self.train_accs,
                "val_losses": self.val_losses,
                "val_accs": self.val_accs
            }, args.checkpoint_path.replace('.pt', f"_{epoch}.pt"))


    def load_ckpts(self,ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epochs = ckpt["epoch"]

        self.train_losses = ckpt["train_losses"]
        self.train_accs = ckpt["train_accs"]
        self.val_losses = ckpt["val_losses"]
        self.val_accs = ckpt["val_accs"]