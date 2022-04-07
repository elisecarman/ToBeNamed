#This file contains the code that defines our model, which depends on modules
#in modules.py

import os.path as osp

import torch

torch.manual_seed(17)

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from modules import ImageEncoder, LanguageEncoder
from utils import augment_image


"""resource on model embeddings: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html"""

class PromptLearner(nn.Module):



    def __init__(self, vocab_size, embedding_dim):
        super(PromptLearner, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        """ self.image_encoder = ImageEncoder()
        self.text_encoder = LanguageEncoder() """
        #Question: Where to include activation? (Relu)


    #Takes in a batch of images
    def forward(self, vis_features_first, vis_features_second, inputs_first, inputs_second, image_encoder, text_encoder):

        #Do this step previous to calling the model and its forward pass
        #Code chunk moved to training
        """ print("Augmenting Images")
        batch_aug1, batch_aug2 = augment_image(image_batch)
        print("Encoding Augmented Images (First)")
        encoded_features1 = image_encoder.forward(batch_aug1)
        print("Encoding Augmented Images (Second)")
        encoded_features2 = image_encoder.forward(batch_aug2) """

        print("Collecting Embeddings by Class (First)")
        embeds_first = self.embeddings(inputs_first).view((1, -1))
        embeds_second = self.embeddings(inputs_second).view((1, -1))

        print("Adding Visual Features")
        out_first = np.add(embeds_first, vis_features_first)
        out_second = np.add(embeds_second, vis_features_second)

        print("Attach Prompt")
        FIXED_PROMPT = "A photo of a"

        out_first = FIXED_PROMPT + out_first
        out_second = FIXED_PROMPT + out_second

        
        print("Encoding Prompts")
        with torch.no_grad():
            out_first = text_encoder.forward(out_first)
            out_second = text_encoder.forward(out_second)

        #do clip ZSL prediction using zsl_clip.py
        #depending on prediction, add image features to class vector/prompt (this is the thing we learn)
        #feed prompts into text encoder (the prompt is the static thing "a photo of a" + the <image feature + class rep> vector)
        #do contrastive loss
        """ encoded_text = clip.encode_text(text)
        return encoded_text """

        return (out_first, out_second)