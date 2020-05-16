"""
Training code for GONet DCGAN
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
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from GO_DataSet import GONetDataSet, Normalize, display_num_images

# ********* Change these paths for your computer **********************
DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/GO_Data"
SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints/DCGAN"
USE_GPU = False
DTYPE = torch.float32
WORKERS = 1  # number of threads for Dataloaders
IMAGE_SIZE = 128

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print(f"using device: {device}")


def train_DCGAN(gen, dis, opitmizer, data_loader, epochs=1, save_checkpoints=None):
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
		for t, (x, y) in enumerate(data_loader):
			gen.train()
			dis.train()
	pass


def load_feature_extraction_data(root_path, batch_size):
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
	# train_pos = GONetDataSet(root_path, "train", transform=transform)
	val_pos = GONetDataSet(root_path, "vali", transform=transform)
	test_pos = GONetDataSet(root_path, "test", transform=transform)
	data_sets = [val_pos, test_pos]#[train_pos, val_pos, test_pos]
	display_num_images(data_sets)

	# Create DataLoaders for the data splits
	# loader_train = DataLoader(train_pos, batch_size=batch_size,
	# 						  sampler=RandomSampler(train_pos), num_workers=WORKERS)
	loader_val = DataLoader(val_pos, batch_size=batch_size,
							sampler=RandomSampler(val_pos), num_workers=WORKERS)
	loader_test = DataLoader(test_pos, batch_size=batch_size,
							 sampler=RandomSampler(test_pos), num_workers=WORKERS)
	data_loaders = {"val": loader_val, "test": loader_test}# {"train": loader_train, "val": loader_val, "test": loader_test}
	return data_loaders, data_sets


def main():
	"""Run training of DCGAN"""

	# Hyper parameters
	beta1 = 0.5  # For ADAM optimizer
	batch_size = 64
	nz = 100  # size of latent vector z
	num_epochs = 1

	data_loaders, data_sets = load_feature_extraction_data(DATA_PATH, batch_size)

	# plot some training examples
	real_batch = next(iter(data_loaders["test"]))
	plt.figure(figsize=(8, 8))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2,
											 normalize=True).cpu(), (1, 2, 0)))


if __name__ == "__main__":
	main()