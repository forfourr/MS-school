import albumentations as A
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from SegDataset_01 import customVOCSegmentation
from argparse import ArgumentParser
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from learning_01 import SegLearner


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = A.Compose([
        A.Resize(520,520),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])
    valid_transforms = A.Compose([
        A.Resize(520,520),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])


    ####### argparse 사용 -> 인자 지정 #######
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='computervision/sementation/data',
                        help='데이터셋 파일 경로')
    parser.add_argument('--checkpoint_path', type=str,default='computervision/sementation/weights/deeplabv3_resnet101.pt',
                        help='중간 저장 model 값 저장 경로')
    
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4,
                        help=' default=0인 이유는, 사용자 환경에 따라 달라질 수 있기 때문')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--resume', action='store_true',
                        help='학습의 재개 여부, store_true가 되면 ')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume_epoch', type=int, default=0)


    args = parser.parse_args()



    train_dataset = customVOCSegmentation(args.data_path,
                                          mode='train', transforms=train_transforms)
    valid_dataset = customVOCSegmentation(args.data_path,
                                          mode='val', transforms=valid_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False, drop_last=False)
    


    ####### model 선언 #######
    num_classes = 21
    model = deeplabv3_resnet101(pretrained=True)
    #model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    optimizer= Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()



    ####### 학습
    learner = SegLearner(model, optimizer, criterion)
    learner.train(train_loader, valid_loader,args)

    # 학습을 재개할 때
    if args.resume:
        learner.load_ckpts(args.checkpoint_path.replace('.pt',
                                                        f"{args.resume_epoch}.pt"))