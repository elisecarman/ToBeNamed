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

#This class was taken with few modifications from CSP code
class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16):
        super(CustomTextEncoder, self).__init__()
        self.dtype = dtype

        self.clip_model = clip_model
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

    def tokenize(self, text):
        return torch.cat([clip.tokenize(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, text, class_embeddings, enable_pos_emb=False):
        """The forward function to compute representations for the prompts.
        OLD PARAMETERS:
        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.
        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        #make enough prompts to be filled by each embedding
        #token_ids should have shape [n x l], where
        #n = num of classes, l = length of sequence (defaults to 77 for clip)
        token_ids = clip.tokenize(text).repeat(len(class_embeddings.weight.data), 1)

        eos_idx = int(token_ids[0].argmax()) #find the eos idx

        #get embeddings for the prompt. note that the last word of prompt is
        #dummy word "X", and this will be replaced
        text_embedding = self.token_embedding(token_ids)

        #replace dummy word with class embedding, for each class
        #this loop could be eliminated if we use a vector of idxs
        for idx, class_e in enumerate(class_embeddings.weight.data):
            print("text embed")
            print(text_embedding.shape)
            print("class_e")
            print(class_e.shape)
            text_embedding[idx, eos_idx - 1, :] = class_e.type(self.clip_model.dtype)

        text_features = text_embedding.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        x = self.transformer(x) #on cpu, running into error here
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        tf = (
            x[
                torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
            ]  # POS of <EOS>
            @ self.text_projection
        )
        return tf

#Image encoder simply uses clip
class ImageEncoder(nn.Module):

    def __init__(self, clip_model):
        super(ImageEncoder, self).__init__()
        self.clip_model = clip_model

    def forward(self, text):
        encoded_image = self.clip_model.encode_image(text)
        return encoded_image