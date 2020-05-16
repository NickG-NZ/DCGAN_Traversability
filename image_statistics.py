import os
import sys

import numpy as np
from PIL import Image


DATA_FOLDER = "GO_Data"
data_folder_abs_path = os.path.join(os.abs_path("."), DATA_FOLDER)


def folder_paths(data_folder_path):
	"""
	Return a dictionary of folder paths
	inside the main data_folder
	"""
	img_folder_paths = {}
	for item in os.list_dir(data_folder_path):
		item_path = os.path.join(data_folder_path, item)
		if os.isdir(item_path):
			# We are only intersted in folders
			





def count_files(abs_path):
	"""
	Counts the number of files in a given folder
	with absolute path given by abs_path
	"""
	num_files = 0
	for item in os.list_dir(abs_path):
		if os.isdir(os.path.join(abs_path, item)):
			continue  # We are counting files not folders
		num_files += 1
	return num_files

d