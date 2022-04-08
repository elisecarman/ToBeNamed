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

        self.transformer = clip_model.transformer.cuda()
        self.positional_embedding = clip_model.positional_embedding.cuda()
        self.ln_final = clip_model.ln_final.cuda()
        self.text_projection = clip_model.text_projection.cuda()
        self.token_embedding = clip_model.token_embedding.cuda()

    def tokenize(self, text):
        return torch.cat([clip.tokenize(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text).cuda()
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    #TODO: Make sure we have the <eos> token
    def forward(self, token_ids, class_embeddings, enable_pos_emb=False):
        """The forward function to compute representations for the prompts.
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
        text_embedding = self.token_embedding(token_ids)

        prompt_features = torch.cat((text_embedding, torch.unsqueeze(class_embeddings.weight.data, 2)), 2)
        
        text_features = text_features.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
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