import os.path as osp

import torch

torch.manual_seed(17)

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from torchvision import datasets
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class LanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(LanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs, visual_features, textEncoder):
        #look up embeddings
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)

        out = numpy.add(embeds, visual_features)

        out = get_text_features(out, textEncoder)

        return out


def augment_image(dbatchsize):
    augmented1 = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           # Add random transformations to the image.
                           transforms.RandomAffine(
                               degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                               shear=(-30, 30, -30, 30)),

                           transforms.ToTensor(),
                           """ transforms.Normalize((0.1307,), (0.3081,)) """
                       ])), batch_size = batchsize)

    augmented2 = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           # Add random transformations to the image.
                           transforms.RandomAffine(
                               degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                               shear=(-30, 30, -30, 30)),

                           transforms.ToTensor(),
                           """ transforms.Normalize((0.1307,), (0.3081,)) """
                       ])), batch_size = batchsize)

    return augmented1, augmented2

def preprocess_data():

def get_visual_features():


def get_text_features():


def train(dataloader1, dataloader2, model, loss_fn, optimizer, word_to_ix):
    total_size = dataloader1.shape[0]
    total_loss = 0
    for x in range (total_size):

        # Step 0: visually encode the augmented images

        visual_features = get_visual_features()
        second_visual_features = get_visual_features()

        #  Step 0.5: predict the class of the images with CLIP

        predict_class = ...

       # Step 1.
        # Prepare the inputs to be passed to the model (i.e, turn the words
       # into integer indices and wrap them in tensors)

       word_idxs = torch.tensor([word_to_ix[w] for w in predict_class], dtype=torch.long)
        

       # Step 2. Recall that torch *accumulates* gradients. Before passing in a
       # new instance, you need to zero out the gradients from the old
       # instance
       model.zero_grad()

       # Step 3. Run the forward pass: Add word embeddings to visual features, pass through text encoder.
        text_features = model(dataloader1, visual_features, textEncoder)
        text_features2 = model(inputs, visual_features, textEncoder)

       # Step 4. Compute your loss function. (Again, Torch wants the target
       # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient visual features
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()t text features
        total_loss += loss.item()

    losses.append(total_loss)



def test():

def main():
    #obtain data
    vocab1, vocab2 = augment_image()

    #create model with embedding matrix
    #Here vocab1 and vocab2 should contain both the label and augmented image

    #create a word to index dictionary to peruse embeddings
    #vocab[1]: the labels I think
    word_to_ix = {word: i for i, word in enumerate(vocab1[1])}

    EMBEDDING_DIM = 5
    CONTEXT_SIZE = 3

    losses = []

    model = LanguageModeler(vocab1.shape[0], EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
""" info on loss: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss"""
    loss_func = losses.NTXentLoss(temperature=0.07, **kwargs)


    loss = loss_func(embeddings, labels)

    #call train
    for epoch in range(10):
        train()


if __name__ == '__main__':
    main()
