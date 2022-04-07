#NOTE: This file was taken with limited modifications from 
#COOP - https://github.com/KaiyangZhou/CoOp/blob/main/trainers/zsclip.py

#Right now, this runs ZSL Clip on static image of a dog, with three static
#prompts

import torch
import torch.nn as nn

from clip import clip
from clip.model import convert_weights
from PIL import Image

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
    def build_model(self):
        # cfg = self.cfg
        # classnames = self.dm.dataset.classnames

        print("Loading CLIP")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        
        self.prompt_template = "a photo of a {}"
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device

    #embeddings is a list of embeddings for each class name
    def model_inference(self, images, embeddings):
        prompts = [prompt_template.format(e) for e in embeddings]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        image = self.preprocess(images).unsqueeze(0).to(self.device)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    #for these two functions, might have to check the output dim
    def encode_image(image):
        return self.clip_model.encode_image(image)

    def encode_text(text, embedding=None):
        if embedding:
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            promts = torch.cat(prompts, 0) #add class embedding to the end of prompt
            return self.clip_model.encode_text(prompts)
        else:
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            return self.clip_model.encode_text(prompts)


