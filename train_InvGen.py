"""
Training code for GONet DCGAN
@author Nick Goodson
"""
import sys
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
import torchvision.transforms as T
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from DataSetDCGAN import DataSetDCGAN
from GONet_torch import Generator, Discriminator, InvGen, weights_init
from utils import Normalize, display_num_images, save_model_params, load_model_params

# ********* Change these paths for your computer **********************
DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/GO_Data"
SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints/Autoencoder"
LOAD_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints/DCGAN"
GEN_FILE = "gen_params__loss0.89127_epoch0"
DIS_FILE = "dis_params__loss1.40223_epoch0"
USE_GPU = True
DTYPE = torch.float32
WORKERS = 0  # number of threads for Dataloaders (0 = singlethreaded)
IMAGE_SIZE = 128
# RANDOM_SEED = 291
PRINT_EVERY = 20  # num iterations between printing learning stats
SAVE_EVERY = 50  # num iterations to save weights after (Assign to 'None' to turn off)
SAVE_TIME_INTERVAL = 60 * 20  # save model every 20 minutes
TEST_GEN_EVERY = 30  # how often (in iterations) to check images created by Gen(InvGen(I))

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print(f"using device: {device}")

# torch.manual_seed(RANDOM_SEED)


# Hyper parameters
########################################
beta1 = 0.9  # Different to GAN
batch_size = 64
num_epochs = 1
lr_invgen = 0.0004
lmbda = 0.5  # tuning parameter for invgen loss function L(z) = (1-lmbda)*Lr + lmbda*Ld
			# Lr = ||I - Gen(z)||, Ld = ||f(I) - f(Gen(z))||
#######################################


def train_invgen(invgen, gen, dis, optimizer, loss_fn, loader_train, loader_val):
	"""
	Training the InvGen network for the autoencoder on GONet data-set using Pytorch.
	Training is done on all positive examples from the data-set

	Inputs:
	- invgen: the inverse generator (autoencoder) network
	- gen: The generator network
	- dis: The discriminator network
	- optimizer: Optimizer object for invgen
	- loss_fn: The loss function to optimize
	- loader_train: Dataloader object for training data
	- loader_val: Dataloader object for validation data

	Returns:
	- gen_test_imgs: a list of grids of images generated by gen during training
	- loss_hist: loss history of autoencdoer invgen->gen network
	"""
	# Create batch of images to visualize
	# the progression of the autoencoder
	fixed_images = next(iter(loader_val))
	gen_test_imgs = []
	loss_hist = []

	invgen = invgen.to(device=device)
	gen = gen.to(device=device)
	dis = dis.to(device=device)
	invgen.train()  # put the networks in training mode
	gen.train()
	dis.train()

	tic = time.perf_counter()  # start timer for model saves
	timed_save = False
	print("\nStarting Training of InvGen")
	for e in range(num_epochs):
		for t, (x, y) in enumerate(loader_train):
			x = x.to(device=device, dtype=DTYPE)
			invgen.zero_grad()
			dis.zero_grad()
			gen.zero_grad()

			z = invgen(x)  # create batch of latent vectors from batch of images
			fake_imgs = gen(z)
			dis_features_fake = dis(fake_imgs)  # last conv layer of dis
			dis_features_real = dis(x)
			loss = loss_fn(x, fake_imgs, dis_features_fake, dis_features_real)

			# Perform update
			loss_hist.append(loss.item())
			loss.backward()
			optimizer.step()

			if t % PRINT_EVERY == 0:
				# Calculate performance stats and display them
				im_residuals = abs(x - fake_imgs).sum().item()
				print(f"Epoch: {e}/{num_epochs}\t Iteration: {t}\n"f"InvGen loss: {loss_hist[-1]:.3f}"
				      f" Image residuals (I - I'): {im_residuals:.3f}")

			if (t + 1) % TEST_GEN_EVERY == 0 or ((e == num_epochs - 1) and t == len(loader_train) - 1):
				# Create images using gen to visualize progress
				with torch.no_grad():
					z = invgen(fixed_images).detach()
					ims = gen(z).detach()
				gen_test_imgs.append(vutils.make_grid(ims, padding=2, normalize=True))
				# *** uncomment to see images produced by gen during training (unlikely to work on remote server) ***
				# plt.imshow(np.transpose(gen_test_imgs[-1].cpu(), (1, 2, 0)))
				# plt.show()

			# Check if time to save model
			toc = time.perf_counter()
			time_diff = toc - tic
			if time_diff > SAVE_TIME_INTERVAL:
				tic = time.perf_counter()  # reset clock
				timed_save = True
				print("Timed save")

			if (SAVE_EVERY and (t + 1) % SAVE_EVERY == 0) or timed_save:
				# Save the model weights in a folder labelled with the validation accuracy
				save_model_params(invgen, "invgen", SAVE_PATH, e, loss_hist[-1])
				timed_save = False

	return gen_test_imgs, loss_hist


def autoencoder_loss(imgs, fake_imgs, dis_features_fake, dis_features_real):
	"""
	The autoencoder loss function described in the GONet paper

	Inputs:
	- imgs: the real batch of images
	- fake_imgs: the images generated by Gen from the latent vectors z output by InvGen
	- dis_features_fake: the last conv layer outputs of Dis for the fake images
	- dis_features_real: "    " for the real images
	- lmbda: A hyper parameter to tune how much of the loss comes from the img residuals ||I - I'||
				and how much from Dis features ||f(I) - f(I')||
	"""
	num_imgs = imgs.size()[0]
	residual_loss = torch.abs(imgs - fake_imgs).sum()
	dis_loss = torch.abs(dis_features_real - dis_features_fake).sum()
	loss = ((1 - lmbda) * residual_loss + lmbda * dis_loss) / num_imgs
	return loss


def load_feature_extraction_data(root_path):
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
	train_pos = DataSetDCGAN(root_path, "train", transform=transform)
	val_pos = DataSetDCGAN(root_path, "vali", transform=transform)
	test_pos = DataSetDCGAN(root_path, "test", transform=transform)
	print("Loaded Automatically Labelled, Positive Datasets")
	data_sets = [train_pos, val_pos, test_pos]
	display_num_images(data_sets)

	# Create DataLoaders for the data splits
	loader_train = DataLoader(train_pos, batch_size=batch_size, drop_last=True,
							  sampler=RandomSampler(train_pos), num_workers=WORKERS)
	loader_val = DataLoader(val_pos, batch_size=batch_size, drop_last=True,
							sampler=RandomSampler(val_pos), num_workers=WORKERS)
	loader_test = DataLoader(test_pos, batch_size=batch_size, drop_last=True,
							 sampler=RandomSampler(test_pos), num_workers=WORKERS)
	data_loaders = {"train": loader_train, "val": loader_val, "test": loader_test}
	return data_loaders, data_sets


def main():
	"""Run training of DCGAN"""
	data_loaders, data_sets = load_feature_extraction_data(DATA_PATH)

	# plot some training examples
	plot_examples = False
	if plot_examples:
		real_batch = next(iter(data_loaders["train"]))
		plt.figure(figsize=(8, 8))
		plt.axis("off")
		plt.title("Training Images")
		plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2,
		                                         normalize=True).cpu(), (1, 2, 0)))
		plt.show()

	# Create the InvGen network
	invgen = InvGen()
	invgen.apply(weights_init)

	# Load the Generator and Discriminator networks
	gen = Generator()
	gen, _, _ = load_model_params(gen, os.path.join(LOAD_PATH, GEN_FILE), device)
	for param in gen.parameters():
		param.requires_grad = False  # freeze model

	dis = Discriminator()
	dis, _, _ = load_model_params(dis, os.path.join(LOAD_PATH, DIS_FILE), device)
	dis.mode = "eval"  # outputs conv layers instead of binary class
	for param in dis.parameters():
		param.requires_grad = False  # freeze model

	# Set up for training
	optimizer = optim.Adam(invgen.parameters(), lr=lr_invgen, betas=(beta1, 0.999))

	# Train the DCGAN network
	gen_test_imgs, loss_hist = \
		train_invgen(invgen, gen, dis, optimizer, autoencoder_loss, data_loaders["train"], data_loaders["val"])

	# Plot loss
	plt.figure(figsize=(10,5))
	plt.title("Autoencoder training loss")
	plt.plot(loss_hist)
	plt.xlabel("iteration")
	plt.ylabel("loss")
	plt.show()

	# Plot some real images
	real_batch = next(iter(data_loaders["train"]))
	plt.figure(figsize=(15, 15))
	plt.subplot(1, 2, 1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2,
	                                         normalize=True).cpu(), (1, 2, 0)))
	# Plot the fake images from the last epoch
	plt.subplot(1, 2, 2)
	plt.axis("off")
	plt.title("Fake Images")
	plt.imshow(np.transpose(gen_test_imgs[-1], (1, 2, 0)))
	plt.show()

if __name__ == "__main__":
	main()
