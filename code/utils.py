import os.path as osp
import torch
torch.manual_seed(17)

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

def augment_image(preprocess_transform, batchsize):
    augmented1 = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.Compose([
                        transforms.RandomAffine(
                            degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                            shear=(-30, 30, -30, 30)),
                        preprocess_transform
                    ])), batch_size = batchsize)

    augmented2 = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform= transforms.Compose([
                        transforms.RandomAffine(
                            degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                            shear=(-30, 30, -30, 30)),
                        preprocess_transform
                    ])), batch_size = batchsize)

    return (augmented1, augmented1) #temporary edit to see if different transform fixes problem