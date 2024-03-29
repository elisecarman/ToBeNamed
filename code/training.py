import os.path as osp

import torch

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from torchvision import datasets
from zsl_clip import ZeroshotCLIP
from PIL import Image
from utils import augment_image
from model import PromptLearner

def train(augmented1, augmented2, model, loss_fn, optimizer, image_encoder, text_encoder):
    total_size = len(augmented1)
    total_loss = 0

    #need to fix this for loop statement
    for i, (images, labels) in enumerate(augmented1):

        # Step 0: visually encode the augmented images

        visual_features = image_encoder.forward(augmented1)
        second_visual_features = image_encoder.forward(augmented2)

        #  Step 0.5: predict the class of the images with CLIP

        predict_class_first = ...
        predict_class_second = ...

       # Step 1.
        # Prepare the inputs to be passed to the model (i.e, turn the words
       # into integer indices and wrap them in tensors)

        word_to_ix = {
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

       # Step 2. Recall that torch *accumulates* gradients. Before passing in a
       # new instance, you need to zero out the gradients from the old
       # instance
        model.zero_grad()

       # Step 3. Run the forward pass: Add word embeddings to visual features, pass through text encoder.

        text_features1, text_features2 = model(
            augmented1, 
            augmented2, 
            word_idxs_first, 
            word_idxs_second, 
            text_encoder)

       # Step 4. Compute your loss function. (Again, Torch wants the target
       # word wrapped in a tensor)
        loss = loss_function(text_features1, text_features2, dtype=torch.long))
        """loss = loss_func(embeddings, labels)"""

        # Step 5. Do the backward pass and update the gradient visual features
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()t text features
        total_loss += loss.item()

    return (total_loss / total_size)

def main():
    #obtain data
    augmented1, augmented2 = augment_image()
    #augmented1 and augmented2 shape: (60000/BATCHSIZE,  [64, 1, 28, 28](images), 64(labels))

    EMBEDDING_DIM = 5
    VOCAB_SIZE = 10
    #create model with embedding matrix
    model = PromptLearner(VOCAB_SIZE, EMBEDDING_DIM)

    #Keeping this here, but will probably not need it
    """ 
    #create a word to index dictionary to peruse embeddings
    word_to_ix = {word: i for i, word in enumerate(labels)} """

    
    losses = []

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # info on loss: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss
    loss_func = losses.NTXentLoss(temperature=0.07, **kwargs)

   image_encoder = ...
   text_encoder = ...

    #call train
    for epoch in range(10):
        loss = train(augmented1, augmented2, model, loss_func, optimizer,image_encoder, text_encoder)
        losses += loss


def main2():
    print("hello")
    clip = ZeroshotCLIP()
    clip.build_model()
    clip.model_inference()


if __name__ == '__main__':
    main2()
