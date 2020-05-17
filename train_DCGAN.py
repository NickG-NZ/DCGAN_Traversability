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
PRINT_EVERY = 20  # num epochs between printing learning stats

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print(f"using device: {device}")

torch.manual_seed(RANDOM_SEED)


def train_dcgan(gen, dis, optimizer_g, optimizer_d, loss_fn, data_loader,
				batch_size, nz, epochs=1, save_checkpoints=None):
	"""
	Training the DCGAN on GONet data-set using Pytorch.
	Training is done on all positive examples from the data-set
	The generator Gen(z) images are labelled with 0.0 (or 0.1)
	The true images are labelled with 1.0 (or 0.9)

	Inputs:
	- gen: The generator network
	- dis: The discriminator network
	- optimizer_g: Optimizer object for gen
	- optimizer_d: Optimizer object for dis
	- loss_fn: The loss function to optimize
	- data_loader: Dataloader object for training data
	- batch_size
	- nz: size of the latent vector z used for the generator
	- epochs: (optional) integer giving number of epochs to train for
	- save_checkpoints: (optional) integer giving frequency of saves in epochs

	Returns:
	- nothing, displays accuracy during training and saves model.
	"""
	# Create batch of latent vectors to visualize
	# the progression of the generator
	fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

	gen_loss_hist = []
	dis_loss_hist = []
	gen_acc_hist = []
	dis_acc_hist = []

	gen = gen.to(device=device)
	dis = dis.to(device=device)
	gen.train()  # put the networks in training mode
	dis.train()
	print("Starting Training of DCGAN")
	for e in range(epochs):
		for t, (x, y) in enumerate(data_loader):
			x = x.to(device=device, dtype=DTYPE)

			# Perform Dis training update ( minimize -ylog(D(x)) - (1-y)log(1-D(G(z)) )
			# 1) Classify a real data batch using Dis
			dis.zero_grad()

			# create labels for real images w/ one-sided label smoothing. (This should technically be a 1.0)
			labels = torch.full((batch_size, 1), 0.9, device=device, dtype=DTYPE)
			logits = dis(x)  # real_scores
			loss_dis_real = loss_fn(logits, labels)
			loss_dis_real.backward()  # compute gradients

			# 2) Classify a fake data batch (from Gen) using Dis
			z = torch.randn(batch_size, nz, 1, 1, device=device)
			fake_imgs = gen(z)
			labels.fill_(0.0)  # create labels for fake images (no smoothing)
			logits = dis(fake_imgs.detach())  # detach() stops gradients being propagated through gen()
			loss_dis_fake = loss_fn(logits, labels)
			loss_dis_fake.backward()

			# 3) Adam update on Dis
			dis_loss_hist.append(loss_dis_real + loss_dis_fake)
			optimizer_d.step()

			# Perform Gen training update ( minimize -log(D(G(z)) )
			gen.zero_grad()
			labels.fill_(1.0)  # label swap trick
			logits = dis(fake_imgs)
			loss_gen = loss_fn(logits, labels)
			loss_gen.backward()
			optimizer_g.step()

	# TODO: write function to calculate performance on val data


def sigmoid_cross_entropy_loss(logits, targets):
	"""
	Calculates the binary cross-entropy loss for a
	batch of logits with respect to targets.
	The batch dimension of targets must be the same as logits

	Inputs:
	- logits: un-normalized outputs from network (batch, 1)
	- targets: targets corresponding to logits (batch, 1)
	"""
	assert (logits.size() == targets.size()), "Logits and targets must have same dimensions"
	probs = torch.sigmoid(logits)
	loss = -targets * torch.log(probs) - (1 - targets) * torch.log(torch.ones_like(probs) - probs)
	return loss


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
	# Positive automatically labelled data
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
	optimizer_g = optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta1, 0.999))
	optimizer_d = optim.Adam(dis.parameters(), lr=lr_dis, betas=(beta1, 0.999))

	# can't use binary loss because Dis has been designed to output two values
	# rather than one.
	loss = nn.BCEWithLogitsLoss

	# Train the DCGAN network
	train_dcgan(gen, dis, optimizer_g, optimizer_d, loss, data_loaders["train"],
				batch_size, nz, epochs=1, save_checkpoints=None)


if __name__ == "__main__":
	main()
