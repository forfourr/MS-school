import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg11, VGG11_Weights

 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset_01 import CustomDataset
import torch.optim as optim
from torch.optim import AdamW

def train(model, train_loader,val_loader,epochs,device, optimizer, criterion):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs =[]

    for epoch in range(epochs):

        ## train
        print('Start training')
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # acc 계산
            _, predicted = torch.max(output.data, 1)
            train_acc += (predicted==target).sum().item()
            # loss
            train_loss += loss.item()
            if i %10 ==9:
                print(f"Epoch: [{epoch+1}/{epochs}], Loss: {loss.item()}")

        
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader.dataset))
        

        ## eval
        print("Start Evaluation")
        model.eval()
        val_loss =0.0
        val_acc = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                # acc
                _, pred = torch.max(output.data,1)
                val_acc +=(pred == target).sum().item()
                # loss
                val_loss += criterion(output, target).item()

        val_losses.append(val_loss/len(train_loader))
        val_accs.append(val_acc/len(val_loader.dataset))

        ## save model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'model.pth')

    return model, train_losses, val_losses, train_accs, val_accs


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #data_path ="C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data"
    data_path = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
    #model = vgg11(pretrained=True)
    model = vgg11(weights=VGG11_Weights.DEFAULT)
    # 마지막 아웃풋 바꿔주기 위해
    # (6): Linear(in_features=4096, out_features=1000, bias=True)
    num_feature = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_feature, 3)
    model.to(DEVICE)


    ##trainfroms
    """
    aug(rezie,aug,flip) -> totensor -> noralization
    """
    train_transfrom = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    val_transfrom = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    ## dataset
    train_dataset = CustomDataset(f'{data_path}/sound/train',transform=train_transfrom)
    val_dataset = CustomDataset(f'{data_path}/soound/val',transform=val_transfrom)
    ## dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)


    # ### 제일 빠른 num_work찾기
    # import time
    # import math

    # test = time.time()
    # math.factorial(100000)
    # for data, t in train_loader:
    #     print(data,t)
    # test01 = time.time()
    # print(f"{test01 - test:.5f} sec")


    ## loss, optim
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),lr= 0.001, weight_decay=1e-2 )

    train(model, train_loader,val_loader,epochs=20,device=DEVICE, optimizer=optimizer, criterion=criterion)


if __name__ == "__main__" :
    main()