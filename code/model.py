#This file contains the code that defines our model, which depends on modules
#in modules.py

import os.path as osp

import torch

torch.manual_seed(17)

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from modules import ImageEncoder, LanguageEncoder
from utils import augment_image

class PromptLearner(nn.Module):

    def __init__(self):
        super(PromptLearner, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = LanguageEncoder()

    #Takes in a batch of images
    def forward(self, image_batch):
        print("Augmenting Images")
        batch_aug1, batch_aug2 = augment_image(image_batch)
        print("Encoding Augmented Images (First)")
        encoded_features1 = self.image_encoder.forward(batch_aug1)
        print("Encoding Augmented Images (Second)")
        encoded_features2 = self.image_encoder.forward(batch_aug2)

        

        #do clip prediction
        #depending on prediction, add features to class vector
        #feed prompts into text encoder
        #do contrastive loss
        encoded_text = clip.encode_text(text)
        return encoded_text