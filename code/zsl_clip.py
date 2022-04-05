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

        # clip_model.to(self.device)

        # temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        # prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = ["a photo of a pickup truck", "a photo of a dog", "a photo of an apple"]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text = prompts
        self.text_features = text_features
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device

    def model_inference(self):
        image = self.preprocess(Image.open("data/cockapoo.png")).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        probs = logits.softmax(dim=-1).cpu()
        print("Logits: " + str(logits))
        print("Probs: " + str(probs))
        return logits

