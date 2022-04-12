#NOTE: This file was taken with limited modifications from 
#COOP - https://github.com/KaiyangZhou/CoOp/blob/main/trainers/zsclip.py

#Right now, this runs ZSL Clip on static image of a dog, with three static
#prompts

import torch
import torch.nn as nn

from clip import clip
from clip.model import convert_weights
from PIL import Image
from torchvision import transforms
import numpy as np

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

class ZeroshotCLIP(nn.Module):

    def __init__(self, text_encoder, image_encoder):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super(ZeroshotCLIP, self).__init__()
        self.prompt_template = "a photo of a"
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.device = device

    #embeddings is a list of embeddings for each class name
    def model_inference(self, images, embeddings):
        #Tokenizes the static prompt, once for each class/embedding
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompt_ids = torch.cat([clip.tokenize(self.prompt_template) for _ in range(len(embeddings.weight.data))])
        text_features = self.text_encoder(prompt_ids, embeddings)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        
        #Added
        predicted_classes = int(logits.argmax())

        return logits, predicted_classes

        # prompts = torch.cat([clip.tokenize(self.prompt_template) for _ in range(len(np.array(embeddings.weight.data)))])
        # prompts = torch.cat((prompts, torch.tensor(embeddings.weight.data)), dim=1)
        # logits_per_image, logits_per_text = self.clip_model(images, prompts)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # preds = torch.argmax(probs, dim=1)
        # return preds, probs
    
    #NOTE: Below two fucntions are old and no longer used

    #for these two functions, might have to check the output dim
    def encode_image(self, images):
        return self.clip_model.encode_image(images)

    def encode_text(self, text, embedding=None):
        if embedding:
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            promts = torch.cat(prompts, 1) #add class embedding to the end of prompt
            return self.clip_model.encode_text(prompts)
        else:
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            return self.clip_model.encode_text(prompts)


