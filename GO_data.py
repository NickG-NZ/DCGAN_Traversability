"""
Creating a Pytorch dataset from the GONet image data
@author Nick Goodson
"""

import os

import torch
from torch.utils.data import Dataset

import numpy as np
import time

from PIL import Image
from PIL.ImageOps import mirror


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

	Args:
	- root_dir (string): Absolute path to directory with image folders
	- split (string): "train", "vali" or "test"
	- label (string): "positive" or "mixed"
			(for training feature extractor vs. classifier)
	- transform (callable): Optional transform/s to be applied on a sample
	"""
	def __init__(self, root_dir, split, label="positive", transform=None):
		self.root_dir
		self.split = split
		self.label = label
		self.split_folder = self._split_folder_name()  # (string) top level folder of selected split
		self.split_dir = os.path.join(self.root_dir, self.split_folder)

		self._check_directories_valid()
		self.data_folders = self._get_data_dirs()  # dict of paths for each subfolder
		self.folder_counts = self.num_images_in_folders()  # dict of num imgs in each subfolder
		self.transform = transform

	def __len__(self):
		"""
		Returns the number of images in this instance
		of the dataset
		"""
		num_images = 0
		for _, count in self.num_images_in_folders().items():
			num_images += count
		return num_images

	def __getitem__(self, idx):
		"""
		Creates a mapping between integer idx and
		images in the data_folders

		Inputs:
		- idx: an index from a Sampler corresponding to a particular image

		Returns:
		- (image, label): (tuple) contains the processed image, and label
				label is either 0.0 or 1.0 for negative and positve images respectively 
		"""
		if torch.is_tensor(idx):
			idx_list = idx.tolist()
			idx = idx_list[0]

		# Debugging code (delete later)
		# --------------------
		if len(idx_list) > 1:
			print("idx has more than one item")
			raise ValueError("To stop the program")
		# --------------------------

		# Place the folders in an arbitrary order for extracting images
		# using indexes from 0 to the length of the dataset
		folder_counts = sorted(self.folder_counts.items())
		f_idx = 0
		(folder, count) = folder_counts[f_idx]
		while idx > count:
			idx -= count  # start idx at beginning of next folder
			f_idx += 1
			if f_idx > len(folder_counts)-1:
				raise IndexError(\
					"index requested from the dataset exceeds dataset length")
			(folder, count) = folder_counts[f_idx]
		img_path = self.data_folders[folder]

		# Load image and apply transforms from Compose object
		img = self.transform(Image.open(path))
		label = 1.0 if "positive" in folder else 0.0

		return (img, label)

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

	def _check_directories_valid(self):
		"""
		Check that the given root_directory
		actually contains the GONet dataset
		"""
		assert(self.root_dir.split("/")[-1] == "GO_Data"), \
			"The given root directory does not point to GO_Data"

		sub_folders = os.listdir(self.split_dir)
		assert(len(sub_folders) == 4), \
			"There should be 4 subfolders in the split's directory"

	def _get_data_dirs(self):
		"""
		Returns a dictionary of paths for each of
		the useful sub folders.
		The keys of the dictionary depend on self.label
		"positive" or "mixed"
		"""
		subfolders = {"positive": ["positive_R", "positive_L"],
					"mixed": ["positive_R", "positive_L", "negative_R", "negative_L"]}
		data_folders = {sub: os.path.join(self.split_dir, sub) for sub in subfolders[self.label]}
		return data_folders

	def num_images_in_folders(self):
		"""
		Returns a dict with the number of images in each of
		the folders in self.data_folders
		"""
		folder_counts = {}
		for name, folder_path in self.data_folders.items():
			folder_counts[name] = len(os.listdir(folder_path))
		return folder_counts


class Normalize():
	"""
	An image tranformation.
	Normalizes images to the range [-1, 1]
	"""
	def __call__(self, image):
		"""
		Inputs:
		- image: (torch tensor) the datapoint. image should already be
			normalized in range [0.0, 1.0] by the ToTensor() transform

		"""
		if not is_tensor(image):
			raise TypeError("Normalize takes images as pytorch tensors")
		if image.max() > 1.0 or image.min() < 0.0:
			raise ValueError("Image should have been normalized to range [0.0, 1.0]")
		image *= 2 - 1
		return (image, label)
