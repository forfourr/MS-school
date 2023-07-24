import torch
import torch.nn as nn
import torchvision
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.resnet import resnet50
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

class SportClassifier:
    def __init__(self, model, optimizer, criterion):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

        self.start_epochs = 0

        self.train_losses= []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train(self, train_loader, val_loader,args):
        best_val_acc = 0.0

        print("start train")
        for epoch in range(self.start_epochs, args.epochs):
            train_acc = 0.0
            train_loss = 0.0
            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch: {epoch+1}/{args.epochs}"), leave=False)

            for i, (data, label,_) in enumerate(train_loader_iter):
                data = data.float().to(self.device)
                label = label.to(self.device)
                output = self.model(data)

                self.optimizer.zero_grad()
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(output,1)
                train_acc += (pred == label).sum().item()

                train_loss += loss.item()

            avg_train_acc = train_acc / len(train_loader.dataset)
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
                    for data,label in val_loader:
                        data= data.float().to(self.device) 
                        label = label.to(self.device)

                        output = self.model(data)
                        
                        loss = self.criterion(output,label)
                        _, pred = torch.max(output,1)

                        val_acc += (pred == label).sum().item()
                        val_loss += loss.item()

                avg_val_acc = val_acc/len(val_loader.dataset)
                avg_val_loss = val_loss/len(val_loader)

                self.val_accs.append(avg_val_acc)
                self.val_losses.append(avg_val_loss)

                ################# Save model #################
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
            

            # epoch가 끝나면 파일 저장
            # 매번 epoch마다 epoch num으로 파이 ㄹ젖
            # 만약 매 epoch모두 저장 안할 경우 replace 제거
            torch.save({
                "epoch": epoch,
                "model_state_dict":self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "train_accs": self.train_accs,
                "val_losses": self.val_losses,
                "val_accs": self.val_accs
            }, args.checkpoint_path.replace('.pt', f"_{epoch}.pt"))
            
    #     self.visualize(self.train_losses, self.val_losses)
            

    # def visualize(self, train_losses, val_losses):
    #     plt.figure()
    #     plt.plot(train_losses, label='train loss')
    #     plt.plot(val_losses, label='val loss')
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.legend()
    #     plt.savefig('loss_plot.png')
    #     plt.show()


    # 현재 모델의 파라미터를 가져오거나 저장
    def load_ckpt(self,ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epochs = ckpt["epoch"]

        self.train_losses = ckpt["train_losses"]
        self.train_accs = ckpt["train_accs"]
        self.val_losses = ckpt["val_losses"]
        self.val_accs = ckpt["val_accs"]






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint_path", type=str,
                        default=f"{PATH}/weight/sport_classifier_checkpoint.pt")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default=f"{PATH}/weight")
    # 커멘드 라인에 이 인자가 있을 때 True 들어감
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_epoch", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)



    parser.add_argument("--train_path", type=str,
                        default=f"{PATH}/sport_dataset/train")
    parser.add_argument("--valid_path", type=str,
                        default=f"{PATH}/sport_dataset/valid")
    

    args = parser.parse_args()

    weight_param_path = args.checkpoint_folder_path
    os.makedirs(weight_param_path, exist_ok=True)

    model = resnet50(pretrained=True)
    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 100)

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay = args.weight_decay)

    classifier = SportClassifier(model, optimizer, criterion)

    # 중간 어쩌구 일때 체크포인트 로드
    # 파일 이름에 epoch 수 저장했을 떄, 터미널에서 받아야함
    if args.resume_training:
        classifier.load_ckpt(args.checkpoint_path.replace(".pt",
                                                          f"{args.resume_epoch}.pt"))

    train_transfrom = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2()

    ])
    val_transfrom = A.Compose([
        ToTensorV2()
    ])

    
    train_dataset = Customdataset(args.train_path,'train',transform=train_transfrom)
    val_dataset = Customdataset(args.valid_path,'valid', transform=val_transfrom)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers)

    classifier.train(train_loader, val_loader, args)
    


