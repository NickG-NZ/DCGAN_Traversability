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
from GONet_torch import Generator, Discriminator


# ********* Change these paths for your computer **********************
DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/GO_Data"
SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints/DCGAN"
USE_GPU = False
DTYPE = torch.float32
WORKERS = 0  # number of threads for Dataloaders (0 = singlethreaded)
IMAGE_SIZE = 128
RANDOM_SEED = 222

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print(f"using device: {device}")

torch.manual_seed(RANDOM_SEED)


def train_DCGAN(gen, dis, optimizer, data_loader, hyper_params, epochs=1, save_checkpoints=None, ):
	"""
	Training the DCGAN on GONet data-set using Pytorch.
	Training is done on all positive examples from the data-set
	The generator Gen(z) images are labelled with 0.0 (or 0.1)
	The true images are labelled with 1.0 (or 0.9)

	Inputs:
	- gen: The generator network
	- dis: The discriminator network
	- optimizer: An Optimizer object
	- data_loader: Dataloader object for training data
	- hyper_params: (dict) of hyper-parameters -> {lr_gen, lr_dis, batch_size, beta1, nz}
	- epochs: (optional) integer giving number of epochs to train for
	- save_checkpoints: (optional) integer giving frequency of saves in epochs

	Returns:
	- nothing, displays accuracy during training and saves model.
	"""
	fixed_noise = torch.randn(batch_size, )

	gen = gen.to(device=device)
	dis = dis.to(device=device)
	for e in range(epochs):
		for t, (x, y) in enumerate(data_loader):
			gen.train()
			dis.train()
			x = x.to(device=device, dtype=DTYPE)
			y = y.to(device=device, dtype=DTYPE)



			# TODO: set y = 0.9/0.1


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
		T.RandomHorizontalFlip(p=0.5),  # Flip image horizontally with p % chance
		T.ToTensor(),
		Normalize()])  # Convert images to range [-1, 1]

	# Create data_set objects for each of the data splits
	# Positive data
	train_pos = GONetDataSet(root_path, "train", transform=transform)
	val_pos = GONetDataSet(root_path, "vali", transform=transform)
	test_pos = GONetDataSet(root_path, "test", transform=transform)
	data_sets = [train_pos, val_pos, test_pos]
	print("Loaded Automatically Labelled, Positive Datasets")
	display_num_images(data_sets)

	# Create DataLoaders for the data splits
	loader_train = DataLoader(train_pos, batch_size=batch_size,
							sampler=RandomSampler(train_pos), num_workers=WORKERS)
	loader_val = DataLoader(val_pos, batch_size=batch_size,
							sampler=RandomSampler(val_pos), num_workers=WORKERS)
	loader_test = DataLoader(test_pos, batch_size=batch_size,
							 sampler=RandomSampler(test_pos), num_workers=WORKERS)
	data_loaders = {"train": loader_train, "val": loader_val, "test": loader_test}
	return data_loaders, data_sets


def main():
	"""Run training of DCGAN"""
	# Hyper parameters
	beta1 = 0.5  # For ADAM optimizer
	batch_size = 64
	nz = 100  # size of latent vector z
	num_epochs = 1
	lr_gen = 0.0004
	lr_dis = 0.0001

	data_loaders, data_sets = load_feature_extraction_data(DATA_PATH, batch_size)

	# plot some training examples
	plot_examples = True
	if plot_examples:
		real_batch = next(iter(data_loaders["train"]))
		plt.figure(figsize=(8, 8))
		plt.axis("off")
		plt.title("Training Images")
		plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2,
												 normalize=True).cpu(), (1, 2, 0)))
		plt.show()

	# Create the Generator and Discriminator networks
	gen = Generator()
	dis = Discriminator()

	# Set up for training
	kwargs = {"save_checkpoints": None,

if __name__ == "__main__":
	main()