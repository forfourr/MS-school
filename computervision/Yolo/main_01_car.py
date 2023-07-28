import warnings
warnings.filterwarnings(action='ignore')


import random
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
from customdataset_01_car import Customdataset, collate_fn
from config_01 import config

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ## fix random seed
    ## 재현할 때 같은 값을 내기 위해서
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
    # fix seed
    seed_everything(config['SEED'])

    ### augmentation ###
    def get_train_transforms():
        return A.Compose([
            A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),    # IMG_SIZE=512
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))    # aug를 줘도 bbox에 영향 x


    ### transfrom ###
    def get_test_transforms():
        return A.Compose([
            A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),    # IMG_SIZE=512
            ToTensorV2()
        ])  # test는 boxes,labels 정보가 없음
    
    train_dataset = Customdataset('computervision/Yolo/car_load_dataset/train',
                                  train=True, transforms=get_train_transforms())
    test_dataset = Customdataset('computervision/Yolo/car_load_dataset/test',
                                 train=False, transforms=get_test_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                             shuffle=False)
    

    ### Define model ###
    def build_model(num_classes = config['NUM_CLASS']+1):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        '''(box_predictor): FastRCNNPredictor(
             (cls_score): Linear(in_features=1024, out_features=91, bias=True)
             (bbox_pred): Linear(in_features=1024, out_features=364, bias=True))
        '''
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        '''
        (box_predictor): FastRCNNPredictor(
          (cls_score): Linear(in_features=1024, out_features=35, bias=True)
          (bbox_pred): Linear(in_features=1024, out_features=140, bias=True))
        '''
        return model

    model = build_model() 
    model.to(device)

    ### train ###
    def train(model, train_loader, optimizer, scheduler, device, resume_checkpoint=None):
        best_loss = 999999
        start_epoch = 1

        # 중간에 다시 시작할 때
        if resume_checkpoint is not None:
            checkpoint = torch.load(resume_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch']+1
            print(f"Resuming training from epoch {start_epoch}")


    
    

if __name__ == '__main__':
    main()
