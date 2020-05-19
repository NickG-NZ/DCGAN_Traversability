"""
Training code for GONet classifier
@author Nick Goodson
"""
import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plot

from GO_DataSet import GONetDataSet, Normalize, display_num_images


# ********* Change these paths for your computer **********************
DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/GO_Data"

SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints/classifier"

USE_GPU = False

DTYPE = torch.float32

WORKERS = 1  # number of threads for Dataloaders

IMAGE_SIZE = 128


if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"using device: {device}")


def train_classifier(fc_net, opitmizer, epochs=1, save_checkpoints=None):
    """
    Training the DCGAN on GONet dataset using pytorch

    Inputs:
    - fc_net: the fully-connected classifier network
    - epochs: (optional) integer giving number of epochs to train for
    - save_checkpoint (optional) integer giving frequency of saves during training (in epochs)

    Returns:
    - nothing, displays accuracy during training and saves model.
    """
    gen = gen.to(device=device)
    dis = dis.to(device=device)
    # for e in range(epochs):
    pass


def load_classifier_data(root_path, batch_size):
    """
    Loads the 3 dataset splits for manually labelled positive and negative data
    from GONet dataset

    Inputs:
    - root_path: absolute path to the root folder of the dataset

    Returns:
    - data_loaders: dictionary of pytorch Dataset objects
        {"train":, "test":, "val":}
    """
    transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        Normalize()])

    # Create data_set objects for each of the data splits
    train_mixed = GONetDataSet(root_path, "train", label="mixed", transform=transform)
    val_mixed = GONetDataSet(root_path, "vali", label="mixed", transform=transform)
    test_mixed = GONetDataSet(root_path, "test", label="mixed", transform=transform)
    data_sets = [train_mixed, val_mixed, test_mixed]
    display_num_images(data_sets)

    # Create DataLoaders for the data splits
    loader_train = DataLoader(train_mixed, batch_size=batch_size,
                                    sampler=RandomSampler(train_mixed), num_workers=WORKERS)
    loader_val = DataLoader(val_mixed, batch_size=batch_size,
                                  sampler=RandomSampler(val_mixed), num_workers=WORKERS)
    loader_test = DataLoader(test_mixed, batch_size=batch_size,
                                   sampler=RandomSampler(test_mixed), num_workers=WORKERS)
    data_loaders = {"train": loader_train, "val": loader_val, "test": loader_test}
    return data_loaders, data_sets
