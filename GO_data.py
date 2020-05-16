"""
Creating a pytorch dataset from the GONet images
@author Nick Goodson
"""

import sys
import os
import argparse

import torch
from torch.utils.data import Dataset
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
		self.root_dir
		self.split = split
		self.label = label
		self.split_folder = self._split_folder_name()  # (string) top level folder of selected split
		self.split_dir = os.path.join(self.root_dir, self.split_folder)

		self._check_directories_valid()
		self.data_folders = self._get_data_dirs()  # dict of paths for each label 
		self.transform = transform

	def __len__(self):
		num_images = 0
		for _, count in self.num_images_in_folders().items():
			num_images += count
		return num_images

	def num_images_in_folders(self):
		"""
		Returns the number of images in each of
		the folders in self.data_folders
		"""
		folder_counts = {}
		for name, folder_path in self.data_folders.items():
			folder_counts[name] = len(os.listdir(folder_path))
		return folder_counts

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.)

		im = np.asarray(Image.open(path))

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
		Returns a dictonary of paths for each of
		the useful sub folders.
		The keys of the dictionary depend on self.label
		"positive" or "mixed"
		"""
		subfolders = {"positive": ["positive_R", "positive_L"],
					"mixed": ["positive_R", "positive_L", "negative_R", "negative_L"]}
		data_folders = {sub: os.path.join(self.split_dir, sub) for sub in subfolders[self.label]}
		return data_folders
		


