"""
Creating a pytorch dataset from the GONet images
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

from PIL import Image
from PIL.ImageOps import mirror


DATA_FOLDER_NAME = "GO_Data"





class GONetDataSet(Dataset)
	"""
	GONet traversability dateset
	Contains 6 folders with 4 subfolders each:
		* data_train, data_vali, data_test
			-> positive_L, positive_R, unlabel_L, unlabel_R

		* data_train_annotation, data_vali_annotation, data_test_annotation
			-> positive_L, positive_R, negative_L, negative_R

	This class constructs a Pytorch Dataset object corresponding
	to a single one of the 6 folders listed above.
	The unlabelled data is not used in any dataset objects that are created.
	"""
	def __init__(self, root_dir, split, label="positive", transform=None):
		"""
		Inputs:
		- root_dir (string): Absolute path to directory with image folders
		- split (string): "train", "val" or "test"
		- label (string): "positive" or "mixed"
				(for training feature extractor vs. classifier)
		- transform (callable): Optional transform to be applied on a sample
		"""
		self.split = split
		self.label = label
		self.split_folder = self._split_folder_name()  # (string) top level folder of selected split
		self._check_directories_valid(root_dir)
		self.data_folders = self._get_data_dirs(root_dir)  # dict of paths for each label 

		# self.data_dir = os.path.join(root_dir, split_folder)
		
		self.folder_dirs = _folder_list(root_dir, split_folder)
		self.transform = transform

	def __len__(self):
		num_images = 0
		for folder in self.folder_dirs:
			num_images += len([i for i in os.listdir(folder)])
		return num_images

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.data_dir)

	def _split_folder_name(self):
		"""
		Returns the name of the folder containing
		the selected data split and label
		"""
		if self.label == "positive":
			split_folder = "data_" + self.split
		elif label == "mixed":
			split_folder = "data_" + self.split + "_annotation"
		else:
			raise ValueError("label must be 'positive' or 'mixed'")
		return split_folder

	def _check_directories_valid(self, root_dir):
		"""
		Check that the given root_directory
		actually contains the GONet dataset
		"""
		assert(root_dir.split("/")[-1] == "GO_Data"), \
			"The given root directory does not point to GO_Data"

		sub_folders = os.listdir(self.data_dir)
		assert(len(sub_folders) == 4), "There should be 4 subfolders"

	def _get_data_dirs(self, root_dir):
		"""
		Returns a list of paths for each of
		the useful sub folders.
		"""
		data_folders = []
		subfolders = {"positive": ["positive_R", "positive_L"],
					"mixed": ["positive_R", "positive_L", "negative_R", "negative_L"]}

		for folder in os.listdir(os.path.join(root_dir, self.split_folder)):


	def _folder_list(self):
		"""
		Returns a list of paths for each of
		the useful sub folders
		"""
		subfolders = {"pos": ["positive_R", "positive_L"],
						"mixed": ["positive_R", "positive_L", "negative_R", "negative_L"]}

		if self.split in ["train", "val", "test"]:
			folder_list = [os.path.join(self.data_dir, sub) for sub in subolders["pos"]]

		elif subset in ["train_annotation", "val_annotation", "test_annotation"]:
			folder_list = [os.path.join(self.data_dir, sub) for sub in subolders["mixed"]]
		return folder_list



	def flip_image(img):
		"""Augments an image by flipping it horizontally"""
		return mirror(img)

	def load_image(path):
		im = np.asarray(Image.open(path))


