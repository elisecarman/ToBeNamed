import os.path as osp

import torch

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from pytorch_metric_learning import losses
from torchvision import datasets
from zsl_clip import ZeroshotCLIP
from PIL import Image
from utils import augment_image
from model import PromptLearner
from modules import CustomTextEncoder, ImageEncoder
from clip import clip


def train(augmented1, augmented2, model, loss_fn, optimizer, text_encoder, image_encoder, clip_zsl, device):
    total_size = len(augmented1)
    total_loss = 0

    #need to fix this for loop statement
    dataloader_iterator = iter(augmented2)

    for i, (images, label) in enumerate(augmented1): #NOTE: We won't have the labels
        images = images.to(device)
        label = label.to(device)
        try:
            images2, label2 = next(dataloader_iterator)
            images2 = images2.to(device)
            label2 = label2.to(device)
        except StopIteration:
            dataloader_iterator = iter(augmented_loaded2)
            images2, label2 = next(dataloader_iterator)

        # Step 0: visually encode the augmented images

        with torch.no_grad():
            # print(images)
            # print(type(images))
            visual_features = image_encoder(images)
            second_visual_features = image_encoder(images2)

            #  Step 0.5: predict the class of the images with CLIP
            
            predict_class_first, probs_first = clip_zsl.model_inference(images, model.embeddings)
            predict_class_second, probs_second = clip_zsl.model_inference(images2, model.embeddings)
            #TODO: Delete elements if the predictions are not the same

       # Step 1.
        # Prepare the inputs to be passed to the model (i.e, turn the words
       # into integer indices and wrap them in tensors)

        """   word_to_ix = {
            "T-Shirt" : 0,
            "Trouser" : 1,
            "Pullover" : 2,
            "Dress" : 3,
            "Coat" : 4,
            "Sandal" : 5,
            "Shirt" : 6,
            "Sneaker" : 7,
            "Bag" : 8,
            "Ankle Boot" : 9,
        } 

        word_idxs_first = torch.tensor([word_to_ix[w] for w in predict_class_first], dtype=torch.long)
        word_idxs_second = torch.tensor([word_to_ix[w] for w in predict_class_second], dtype=torch.long)
        """
        
       # Step 2. Recall that torch *accumulates* gradients. Before passing in a
       # new instance, you need to zero out the gradients from the old
       # instance
        model.zero_grad()

       # Step 3. Run the forward pass: Add word embeddings to visual features, pass through text encoder.

        text_features1, text_features2 = model(
            augmented1, 
            augmented2, 
            predict_class_first, 
            predict_class_second)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(text_features1, text_features2, dtype=torch.long)
        """loss = loss_func(embeddings, labels)"""

        # Step 5. Do the backward pass and update the gradient visual features
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()t text features
        total_loss += loss.item()

    return (total_loss / total_size)

def main():
    print("Loading CLIP")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE - {}".format(device))
    clip_model, preprocess_transform = clip.load("ViT-B/32", device=device)
    text_encoder = CustomTextEncoder(clip_model).cuda()
    image_encoder = ImageEncoder(clip_model).cuda()
    clip_zsl = ZeroshotCLIP(text_encoder, image_encoder).cuda()

    classnames = [
        "T-Shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot"
    ]

    #obtain data

    initial_embeddings = torch.cat([clip.tokenize(c) for c in classnames]).float().cuda()
    #create model with embedding matrix
    model = PromptLearner(initial_embeddings, text_encoder).cuda()

    #Keeping this here, but will probably not need it
    """ 
    #create a word to index dictionary to peruse embeddings
    word_to_ix = {word: i for i, word in enumerate(labels)} """

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # info on loss: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss
    loss_func = losses.NTXentLoss(temperature=0.07).cuda()

    loss_list = []

    #call train
    for epoch in range(10):
        print("EPOCH: {}".format(epoch))
        #augmented1 and augmented2 shape: (60000/BATCHSIZE,  [64, 1, 28, 28](images), 64(labels))
        augmented1, augmented2 = augment_image(preprocess_transform, 128) #must augment data each time
        loss = train(augmented1, augmented2, model, loss_func, optimizer, text_encoder, image_encoder, clip_zsl, device)
        loss_list += loss


if __name__ == '__main__':
    main()
