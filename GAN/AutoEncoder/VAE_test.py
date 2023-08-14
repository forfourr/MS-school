from VAE_model import VAE
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

model = VAE()
model.load_state