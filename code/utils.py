import os.path as osp

import torch

torch.manual_seed(17)

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

def augment_image(dbatchsize):
    augmented1 = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           # Add random transformations to the image.
                           transforms.RandomAffine(
                               degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                               shear=(-30, 30, -30, 30)),

                           transforms.ToTensor(),
                           """ transforms.Normalize((0.1307,), (0.3081,)) """
                       ])), batch_size = batchsize)

    augmented2 = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           # Add random transformations to the image.
                           transforms.RandomAffine(
                               degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                               shear=(-30, 30, -30, 30)),

                           transforms.ToTensor(),
                           """ transforms.Normalize((0.1307,), (0.3081,)) """
                       ])), batch_size = batchsize)

    return augmented1, augmented2