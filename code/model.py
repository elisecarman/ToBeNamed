#This file contains the code that defines our model, which depends on modules
#in modules.py

import os.path as osp

import torch

torch.manual_seed(17)

import numpy as np
import torch.nn as nn
import zsl_clip
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from modules import ImageEncoder, LanguageEncoder
from utils import augment_image


"""resource on model embeddings: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html"""

class PromptLearner(nn.Module):



    def __init__(self, initial_embeddings):
        super(PromptLearner, self).__init__()
        #we initialize the embeddings to be the clip embeddings for the classname
        self.embeddings = nn.Embedding(initial_embeddings.size(0), initial_embeddings.size(1))
        print(initial_embeddings.type())
        self.embeddings.weight = nn.Parameter(initial_embeddings)
        self.alpha = 1 #hyperparameter to adjust how much image feature changes prompt
        """ self.image_encoder = ImageEncoder()
        self.text_encoder = LanguageEncoder() """
        #Question: Where to include activation? (Relu)


    #Takes in a batch of images
    def forward(self, vis_features_first, vis_features_second, inputs_first, inputs_second, clip_model):

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
        #depending on prediction, add image features to class vector/prompt (this is the thing we learn)
        out_first = np.add(embeds_first, self.alpha * vis_features_first)
        out_second = np.add(embeds_second, self.alpha * vis_features_second)

        print("Attach Prompt")
        FIXED_PROMPT = "A photo of a"

        #Don't think we want this - fixed prompt is a string that still needs to be
        #turned into embeddings by clip, and out_first/second is already an
        #embedding. This is handled in zsl_clip
        # out_first = FIXED_PROMPT + out_first 
        # out_second = FIXED_PROMPT + out_second

        
        print("Encoding Prompts")
        #NOTE: I'm not positive if this is the way we want to freeze the text encoder,
        #based on this link: https://pytorch.org/docs/master/notes/autograd.html
        with torch.no_grad():
            #feed prompts into text encoder (the prompt is the static thing "a photo of a" + the <image feature + class rep> vector)
            out_first = clip_model.text_encoder(FIXED_PROMPT, embedding=out_first)
            out_second = clip_model.text_encoder(FIXED_PROMPT, embedding=out_second)

        return (out_first, out_second)