"""
General utiities for use in training models
"""
import torch
import os


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


def save_model_params(model, name, save_path, epoch, optimizer=None, final_loss=None):
	"""
	Saves the models parameters to the selected path
	using built in pytorch fucntionality

	Inputs:
	- model: the pytorch model to save
	- name: (string) the name to save the model as
	- save_path: absolute path to save location
	- epoch
	- optimizer
	- final_loss: (optional float) used in the name of the file
	"""
	if final_loss:
		filename = f"{name}_params__loss{final_loss:.5f}_epoch{epoch}.tar"
	else:
		filename = f"{name}_params__loss_NA_epoch{epoch}.tar"
	file_path = os.path.join(save_path, filename)
	save_dict = {'epoch': epoch,
	             'model_state_dict': model.state_dict(),
	             'loss': final_loss}
	if optimizer:
		save_dict['optimizer_state_dict'] = optimizer.state_dict()
	torch.save(save_dict, file_path)
	print(f"Saved model: {name} to file")


def load_model_params(model, path, device, optimizer=None):
	"""
	Loads the model params for both the network and
	the optimizer.

	Inputs:
	- model: the pytorch model object to load the params into
	- path: the full path name of the params file
	- device: the device to load the model onto ('cpu' or 'cuda')
	- optimizer (optional): the optimzier oject to load params into

	Returns:
	- model: the model with params loaded
	- optimizer: the optimizer with params loaded
	- epoch: the epoch at which the params were saved
	- loss: the loss produced by the loaded params
	"""
	# if device == torch.device('cpu'):
	# 	checkpoint = torch.load(path, map_location=device)
	# else:
	checkpoint = torch.load(path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	if optimizer:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		return model, optimizer, epoch, loss
	return model, epoch, loss