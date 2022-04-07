import os.path as osp
import torch
torch.manual_seed(17)

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

def augment_image(batchsize):
    augmented1 = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform= transforms.Compose([
                        transforms.RandomAffine(
                            degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                            shear=(-30, 30, -30, 30)),
                        transforms.ToTensor(),
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
                        transforms.ToTensor(),
                    ])), batch_size = batchsize)

    return(augmented1, augmented2)