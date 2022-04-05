#This file defines the torch modules that make up our model

import os.path as osp

import torch

torch.manual_seed(17)

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from torchvision import datasets
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

#Language encoder simply uses clip
class LanguageEncoder(nn.Module):

    def __init__(self):
        super(LanguageEncoder, self).__init__()

    def forward(self, text):
        text = clip.tokenize(text)
        encoded_text = clip.encode_text(text)
        return encoded_text

#Image encoder simply uses clip
class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()

    def forward(self, text):
        encoded_image = clip.encode_image(text)
        return encoded_image