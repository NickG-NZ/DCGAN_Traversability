"""
Creating a Pytorch dataset for classifier data
Data folder tree must be:
			          -------root--------
		            /         |           \
data_train_annotated  data_vali_annotated  data_test_annotated
   /        \              /       \            /       \
positive  negative    positive  negative    positive  negative

@author Nick Goodson
"""
import os
from torch.utils.data import Dataset
from PIL import Image


class DataSetClassifier(Dataset):
	"""
	Classifier dataset, contains positive and negative examples
	This class constructs a Pytorch Dataset object corresponding to a single one of 3 datasets:
	(train, vali, test)
	Args:
	- root_dir (string): Absolute path to directory with image folders
	- split (string): "train", "vali" or "test"
	- transform (callable): Optional transform/s to be applied on a sample
	"""
	def __init__(self, root_dir, split, transform=None):
		self.root_dir = root_dir
		self.split = split
		self.transform = transform

		self.split_name = "data_" + self.split + "_annotated" # (string) name of folder for selected split
		self.split_dir = os.path.join(self.root_dir, self.split_name) # path to top level folder of selected split
		self.labelled_folders = ["positive", "negative"]
		self.pos_images = sorted(os.listdir(os.path.join(self.split_dir, "positive")))
		self.neg_images = sorted(os.listdir(os.path.join(self.split_dir, "negative")))
		self.pos_len = len(self.pos_images)
		self.neg_len = len(self.neg_images)

	def __len__(self):
		"""
		Returns the number of images in this instance
		of the dataset
		"""
		return self.pos_len + self.neg_len - 2

	def __getitem__(self, idx):
		"""
		Creates a mapping between integer idx and images in the data_folders
		Inputs:
		- idx: an index from a Sampler corresponding to a particular image
		Returns:
		- (image, label): (tuple) contains the processed image, and label is either 0.0 or 1.0
						for negative and positve images respectively
		"""
		# decide if index is for positive or negative folder
		if idx > self.pos_len:
			idx = idx - self.pos_len
			img_name = self.neg_images[idx]
			img_path = os.path.join(self.split_dir, "negative", img_name)
			label = 0.0
		else:
			img_name = self.pos_images[idx]
			img_path = os.path.join(self.split_dir, "positive", img_name)
			label = 1.0

		# Load image and apply transforms from Compose object
		img = Image.open(img_path)
		if self.transform:
			img = self.transform(img)
		return img, label

