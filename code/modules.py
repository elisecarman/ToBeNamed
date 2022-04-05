#This file defines the torch modules that make up our model

import os.path as osp

import torch

torch.manual_seed(17)

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from torchvision import datasets

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