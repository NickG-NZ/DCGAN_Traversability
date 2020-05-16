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
from torch.utils.data import RandomSampler
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plot

from GO_data import GONetDataSet, Normalize, DATA_FOLDER_NAME


# ********* Change these paths for your computer **********************
DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/GO_Data"

SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints"

USE_GPU = False

DTYPE = torch.float32

WORKERS = 1  # number of threads for dataloaders

IMAGE_SIZE = 128


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
	Loads the 3 dataset splits for automatically labelled positive data
	from GONet dataset

	Inputs:
	- root_path: absolute path to the root folder of the dataset
	
	Returns:
	- data_loaders: dictionary of pytorch Dataset objects
		{"train":, "test":, "val":, "train_labelled", "test_labelled", "val_labelled"}
	"""
	transform = T.Compose([
					T.ToTensor(),
					T.RandomHorizontalFlip(),
					Normalize()])

	# Create data_set objects for each of the data splits
	# Positive data
	train_pos = GONetDataSet(root_path, "train", transform=transform)
	val_pos = GONetDataSet(root_path, "vali", transform=transform)
	test_pos = GONetDataSet(root_path, "test", transform=transform)

	# Mixed data
	train_mixed = GONetDataSet(root_path, "train", label="mixed", transform=transform)
	val_mixed = GONetDataSet(root_path, "vali", label="mixed", transform=transform)
	test_mixed = GONetDataSet(root_path, "test", label="mixed", transform=transform)

	# Print the number of images in each folder
	data_sets = [train_pos, val_pos, test_pos, train_mixed, val_mixed, test_mixed]
	data_set_names = ["train_pos", "val_pos", "test_pos", "train_mixed", "val_mixed", "test_mixed"]
	for dataset, name in zip(data_sets, data_set_names):
		print("\n")
		print(f"Dataset: {name}")
		for folder, count in  dataset.folder_counts.items():
			print(f"num images in {folder} folder: {count}")

	# Create DataLoaders for the data splits
	loader_train_pos = DataLoader(train_pos, batch_size=BATCH_SIZE,
									sampler=RandomSampler(train_pos), num_workers=WORKERS)
	loader_val_pos = DataLoader(val_pos, batch_size=BATCH_SIZE,
									sampler=RandomSampler(val_pos), num_workers=WORKERS)
	loader_test_pos = DataLoader(test_pos, batch_size=BATCH_SIZE,
									sampler=RandomSampler(test_pos), num_workers=WORKERS)
	loader_train_mixed = DataLoader(train_mixed, batch_size=BATCH_SIZE,
									sampler=RandomSampler(train_mixed), num_workers=WORKERS)
	loader_val_mixed = DataLoader(val_mixed, batch_size=BATCH_SIZE,
									sampler=RandomSampler(val_mixed), num_workers=WORKERS)
	loader_test_mixed = DataLoader(test_mixed, batch_size=BATCH_SIZE,
									sampler=RandomSampler(test_mixed), num_workers=WORKERS)

	data_loaders = [loader_train_pos, loader_val_pos, loader_test_pos, loader_train_mixed,
					loader_val_mixed, loader_test_mixed]
	return data_loaders, data_sets


def main():
	"""Run training of DCGAN"""
	# Training DCGAN
	# Hyper parameters
	BETA1 = 0.5  # For ADAM optimizer

	# Params for Gen
	BATCH_SIZE_GEN = 64
	NZ = 100  # size of latent vector z

	# Params for classifier
	LR_FC = 0.0001
	BATCH_SIZE_FC = 64

	data_loaders, data_sets = load_data_sets(DATA_PATH)