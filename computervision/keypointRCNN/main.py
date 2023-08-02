import torch
import torchvision
import albumentations as A
import warnings
warnings.filterwarnings(action='ignore')

from engine import train_one_epoch
from utils import collate_fn
from torch.utils.data import DataLoader

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from customdataset import keypointDataset

def get_mode(num_keypoints, weights_path = None)