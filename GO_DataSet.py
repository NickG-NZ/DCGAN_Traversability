"""
Creating a Pytorch dataset from the GONet image data
@author Nick Goodson
"""
import os
import time

import torch
from torch.utils.data import Dataset

from PIL import Image


class GONetDataSet(Dataset):
	"""
	GONet traversability dateset
	Contains 6 folders with 4 subfolders each:
		* data_train, data_vali, data_test
			-> positive_L, positive_R, unlabel_L, unlabel_R

		* data_train_annotation, data_vali_annotation, data_test_annotation
			-> positive_L, positive_R, negative_L, negative_R

	This class constructs a Pytorch Dataset object corresponding to a single one of the 6 folders listed above.
	The unlabelled data is not used in any dataset objects that are created.
	Args:
	- root_dir (string): Absolute path to directory with image folders
	- split (string): "train", "vali" or "test"
	- label (string): "positive" or "mixed"
			(for training feature extractor vs. classifier)
	- transform (callable): Optional transform/s to be applied on a sample
	"""
	def __init__(self, root_dir, split, label="positive", transform=None):
		self.root_dir = root_dir
		self.split = split
		self.label = label
		self.transform = transform

		self.split_name = self._split_folder_name()  # (string) name of folder for selected split
		self.split_dir = os.path.join(self.root_dir, self.split_name) # path to top level folder of selected split
		self._check_directories_valid()
		self.data_folder_paths = self._get_data_dirs()  # dict of paths for each subfolder
		self.folder_counts = self._num_images_in_folders()  # list of tuples, num imgs in each subfolder
		self.image_name_lists = self._get_image_names()
		self.length = self._save_length()  # save length for faster lookup

	def __len__(self):
		"""
		Returns the number of images in this instance
		of the dataset
		"""
		return self.length

	def __getitem__(self, idx):
		"""
		Creates a mapping between integer idx and images in the data_folders
		Inputs:
		- idx: an index from a Sampler corresponding to a particular image
		Returns:
		- (image, label): (tuple) contains the processed image, and label is either 0.0 or 1.0
						for negative and positve images respectively
		"""
		if torch.is_tensor(idx):
			idx = idx.tolist()[0]

		# The folders are in an arbitrary constant order for extracting images
		# using indexes from 0 to the length of the data-set
		f_idx = 0
		(folder, count) = self.folder_counts[f_idx]
		while idx > count:
			idx -= count  # start idx at beginning of next folder
			f_idx += 1
			if f_idx > len(self.folder_counts)-1:
				raise IndexError("index requested from the dataset exceeds dataset length")
			(folder, count) = self.folder_counts[f_idx]
		folder_path = self.data_folder_paths[folder]
		img_name = self.image_name_lists[folder][idx]
		img_path = os.path.join(folder_path, img_name)

		# Load image and apply transforms from Compose object
		img = Image.open(img_path)
		if self.transform:
			img = self.transform(img)
		label = 1.0 if "positive" in folder else 0.0
		return img, label

	def _save_length(self):
		"""
		Calculates and saves the length for fast reference
		in the future
		"""
		num_images = 0
		for _, count in self.folder_counts:
			num_images += count
		return num_images - 1

	def _split_folder_name(self):
		"""
		Returns the name of the folder containing
		the selected data split and label
		"""
		if self.label == "positive":
			split_folder = "data_" + self.split
		elif self.label == "mixed":
			split_folder = "data_" + self.split + "_annotation"
		else:
			raise ValueError("label must be 'positive' or 'mixed'")
		return split_folder

	def _check_directories_valid(self):
		"""
		Check that the given root_directory
		actually contains the GONet dataset
		"""
		assert(self.root_dir.split("/")[-1] == "GO_Data"), "The given root directory does not point to GO_Data"

		sub_folders = os.listdir(self.split_dir)
		assert(len(sub_folders) == 4), "There should be 4 sub-folders in the split's directory"

	def _get_data_dirs(self):
		"""
		Returns a dictionary of paths for each of the useful sub folders.
		The keys of the dictionary depend on self.label (which can be "positive" or "mixed")
		"""
		subfolders = {"positive": ["positive_R", "positive_L"],
					"mixed": ["positive_R", "positive_L", "negative_R", "negative_L"]}
		data_folder_paths = {sub: os.path.join(self.split_dir, sub) for sub in subfolders[self.label]}
		return data_folder_paths

	def _num_images_in_folders(self):
		"""
		Returns a list of tuples with the number of images in each of
		the folders in self.data_folders
		[(folder_name, count), ...]
		"""
		folder_counts = {}
		for name, folder_path in self.data_folder_paths.items():
			folder_counts[name] = len(os.listdir(folder_path))
		folder_counts = sorted(folder_counts.items())  # place folders in arbitrary sorted order
		return folder_counts

	def _get_image_names(self):
		"""
		Saves the names of images in each folder in a list
		for quick reference by self.__getitem__
		Returns a dict of {folder_name: [image_names_list]...}
		"""
		image_name_lists = {}
		for name, path in self.data_folder_paths.items():
			image_name_lists[name] = os.listdir(path)
		return image_name_lists



class Normalize:
	"""
	An image transformation.
	Normalizes images to the range [-1, 1]
	"""
	def __call__(self, image):
		"""
		Inputs:
		- image: (torch tensor) image should already be normalized in range [0.0, 1.0]
				by the ToTensor() transform
		"""
		if not torch.is_tensor(image):
			raise TypeError("Normalize takes images as pytorch tensors")
		if image.max() > 1.0 or image.min() < 0.0:
			raise ValueError("Image should have been normalized to range [0.0, 1.0]")
		image = (image * 2) - 1
		return image


def display_num_images(data_sets):
	"""
	Displays the number of images in each data set in the
	list data_sets

	Inputs:
	- data_sets: (list) of GONetDataSet objects
	"""
	for dataset in data_sets:
		name = dataset.split_name
		print("\n")
		print(f"Dataset: {name}")
		for folder, count in dataset.folder_counts:
			print(f"num images in {folder} folder = {count}")
