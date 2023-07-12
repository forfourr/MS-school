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
resnet_model = resnet18(pretrained=False)    
# CIFAR-10 클래스 개수 10개에 맞게 변경
num_feature_resnet = resnet_model.fc.in_features        # 512
resnet_model.fc = nn.Linear(num_feature_resnet, 10)
# (fc): Linear(in_features=512, out_features=1000, bias=True)
#  (fc): Linear(in_features=512, out_features=10, bias=True)

##########
# Voting 앙상블 모델    
# model, loss, optimc
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=7),
    n_estimators=5
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)  #?????resnet_model param 받는이유


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
def ensemble_predict(models, device, test_loader):
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = []
            model.eval()
            for model in models:
                output = model(data)
                outputs.append(output)

            ensemble_output = torch.stack(outputs).mean(dim=0)
            _, predicted = torch.max(ensemble_output, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions


############
# main
if __name__ == '__main__':
    models = []
    for epoch in range(1,6):
        print(f"Training Model {epoch}")

        model = model.to(device)
        train(model, device, train_loader, optimizer, criterion)

        accuracy = evaluate(model, device, test_loader)
        print(f"Model {epoch} Accuracy: {accuracy:.2f}")
        models.append(model)
        
    ensemble_predictions = ensemble_predict(models, device, test_loader)
    ensemble_accuracy = accuracy_score(test_dataset.targets,ensemble_predictions)
    print(f"Ensemble Acuuracy: {accuracy:.2f}")