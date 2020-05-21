"""
Creating a Pytorch dataset from the GONet image data
@author Nick Goodson
"""
import os
from torch.utils.data import Dataset
from PIL import Image


class DataSetDCGAN(Dataset):
	"""
	GONet DCGAN traversability dateset
	This class constructs a Pytorch Dataset object corresponding to a single one of 3 datasets:
	(train, vali, test)
	Args:
	- root_dir (string): Absolute path to directory with image folders
	- split (string): "train", "vali" or "test"
	- transform (callable): Optional transform/s to be applied on a sample
	"""
	def __init__(self, root_dir, split, transform=None):

		# Check that root_dir points to correct folder
		assert (root_dir.split("/")[-1] == "GO_Data"),\
			"The given root directory does not point to GO_Data"
		self.root_dir = root_dir
		self.split = split
		self.transform = transform

		self.split_name = "data_" + self.split  # (string) name of folder for selected split
		self.split_dir = os.path.join(self.root_dir, self.split_name) # path to top level folder of selected split
		self.images = sorted(os.listdir(self.split_dir))
		self.length = len(self.images) - 1  # number of images
		self.label = 1.0  # all images are positive

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
		img_name = self.images[idx]
		img_path = os.path.join(self.split_dir, img_name)

		# Load image and apply transforms from Compose object
		img = Image.open(img_path)
		if self.transform:
			img = self.transform(img)
		return img, self.label

