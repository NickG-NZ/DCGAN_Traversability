"""
Training code for GONet
@author Nick Goodson
"""
import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.transforms as Training

import numpy as np
import matplotlib.pyplot as plot

from PIL import Image
from PIL.ImageOps import mirror


USE_GPU = False
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print(f"using device: {device}")



def train_DCGAN(gen, dis, opitmizer, epochs=1, save_checkpoints=None):
	"""
	Training the DCGAN on GONet dataset using pytorch

	Inputs:
	- gen: The generator network
	- dis: The discriminator network
	- epochs: (optional) integer giving number of epochs to train for
	- save_checkpoint (optional) integer gving frequency of saves in epochs

	Returns:
	- nothing, displays accuracy during training and saves model.
	"""
	gen = gen.to(device=device)
	dis = dis.to(device=device)
	for e in range(epochs):



def load_data_sets(root_path):
	"""
	Loads the six unique data groups in the GONet
	dataset

	Inputs:
	- root_path: absolute path to the root folder of the dataset
	
	Returns:
	- data_loaders: dictionary of pytorch Dataset objects
		{"train":, "test":, "val":, "train_labelled", "test_labelled", "val_labelled"}
	
	


def main():
	pass