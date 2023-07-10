import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, vgg11
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

 
# Decvice GPU설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device >>", device)


# CIFAR-10 Data
# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# dataset/ loader
train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


# Set VGG/ ResNet model
vgg_model = vgg11(pretrained= False)
resnet_model = resnet18(pretrained=False)
    # 마지막 layer에서 feature 가지고옴
num_feature_vgg = vgg_model.classifier[6].in_features   # 4096
num_feature_resnet = resnet_model.fc.in_features        # 512
# (6): Linear(in_features=4096, out_features=1000, bias=True)
# (fc): Linear(in_features=512, out_features=1000, bias=True)

    # CIFAR-10 클래스 개수 10개에 맞게 변경
vgg_model.classifier[6] = nn.Linear(num_feature_vgg, 10)
resnet_model.fc = nn.Linear(num_feature_resnet, 10)
#  (6): Linear(in_features=4096, out_features=10, bias=True)
#  (fc): Linear(in_features=512, out_features=10, bias=True)

##########
# Voting 앙상블 모델
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__inti__()
        self.models = nn.ModuleList(models)

    def forward(self,x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        avg_outputs = torch.mean(outputs, dim=0)
        return avg_outputs
    
# model, loss, optimc
ensemble_model = EnsembleModel([vgg_model, resnet_model])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ensemble_model.parameters() lr=0.001)


#########
# Train model
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluation
def evaluate(model, device, test_loader):
    model.eval()
    predctions =[]
    targets =[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)
            predctions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy())

    accuarcy = accuracy_score(target, predctions)
    return accuarcy

##########
# 앙상블된 모델 예측 합침
def combine_predictions(predictions):
    combined = torch.cat(predictions, dim=0)
    _, predicted_labels = torch.max(combined, 1)
    return predicted_labels


############
# main
if __name__ == '__main__':
    for epoch in range(1,2):
        print(f"Training Model {epoch}")
        ensemble_model = ensemble_model.to(device)
        train(ensemble_model, device, train_loader, optimizer, criterion)


        predictions = []

        with torch.no_grad():
            for data,_ in test_loader:
                data = data.to(device)
                output = ensemble_model(data)
                predictions.append(output)
        
        combine_predictions = combine_predictions(predictions)
        accuracy = accuracy_score(test_dataset.targets, combine_predictions.cpu().numpy())
        print(f"Model {epoch}, Acuuracy: {accuracy:.2f}")