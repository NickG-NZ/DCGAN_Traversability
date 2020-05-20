"""
General utiities for use in training models
"""
import torch


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
	- data_sets: (list) of DataSet_DCGAN objects
	"""
	for dataset in data_sets:
		name = dataset.split_name
		print("\n")
		print(f"Dataset: {name}")
		print(f"num images = {len(dataset)}")