import torch
import torch.nn as nn
import torchvision.models as models

"""
model = models.vgg11(num_classes=1000)으로 모델을 가지고 오는것은 이미 학습된 모델
class VGG11을 정의한 이유는 내장된 모델의 특징만 추출하기 위해서
분류기를 재정의하면 새로운 작업에 맞게 모델을 사용
"""

class VGG11(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG11, self).__init__()

        #내장 모델에서feature 가져오기 
        self.features = models.vgg11(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x