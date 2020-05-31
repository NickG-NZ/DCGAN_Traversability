"""
Utiltiies for parsing Raw [synced + rectified] Kitti data
Uses the velocity data to automatically classify image frames as traversable / non-traversable
"""

"""
1) Use velocity threshold to separate images into two folders
	- likely traversable/positive
	- likely untraversable/negative
2) Create corresponding data files which hold the velocities (and maybe other measurements)
	for both folders
2) Check the images in the positive folder (along with their velocities) to ensure they seem reasonable

4) Create a script that shows the likely negative images along with their velocity
	and prompts the user for a "y" or "n" (traversable or not) then puts the confirmed non-traversable
	images into a negative test folder (aim for ~400 images)
	and the confirmed positive into a positive test foler (~400 images)

5) Split the main positive folder into train and validation folders (use ~ 80-20 split) in order

6) reduce the size of the images to 128x128 (during loading?)
"""

import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Kitti_Data"
IMG_SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Kitti_Data"
DATA_FOLDERS = ["2011_09_26_drive_0057_sync"]

# Automatically label the data using a velocity threshold
V_THRESH = 0.5  # [m/s]
vf_idx = 8  # index of forward velocity
vl_idx = 9  # index of lateral velocity


def auto_label():
	"""
	Automatically label data based on velocity threshold
	"""
	maybe_make_dir(DATA_PATH, "positive")
	maybe_make_dir(DATA_PATH, "negative")

	for f in range(len(DATA_FOLDERS)):
		oxts_path = os.path.join(DATA_PATH, DATA_FOLDERS[f], "oxts", "data")       # auxiliary data (IMU/GPS)
		rgb1_path = os.path.join(DATA_PATH, DATA_FOLDERS[f], "image_01", "data")  # first RGB camera
		rgb2_path = os.path.join(DATA_PATH, DATA_FOLDERS[f], "image_02", "data")  # second RGB camera

		oxts = os.listdir(oxts_path)
		rgb1_img_list = os.listdir(rgb1_path)
		rgb2_img_list = os.listdir(rgb2_path)
		n_files = len(oxts)

		id_num_prev = -1
		for i in range(n_files):  # loop over all files and sort them using velocity
			file_id = oxts[i].split(".")[0]

			# check that no data is missing (file-names count up from 0)
			try:
				id_num = int(file_id.lstrip("0"))
			except ValueError:  # first id is all zeros
				id_num = 0
			assert(id_num == id_num_prev + 1,
				f"Error: non-continous oxt-data: {id_num_prev}-{id_num}")

			oxtdata = np.genfromtxt(os.path.join(oxts_path, oxts[i]))
			velocity = np.linalg.norm(np.array([oxtdata[vf_idx], oxtdata[vl_idx]])).item()

			# Sort images
			img_name = file_id + ".png"
			imgs = [rgb1_img_list[i], rgb2_img_list[i]]

			for l, im in enumerate(imgs):
				assert (im == img_name,
				        f"couldn't find matching image {l + 1} for id:{file_id}")

			if velocity > V_THRESH:
				# Save corresponding images to the traversable folder
				os.rename(os.path.join(rgb1_path, img_name), os.path.join(DATA_PATH, "positive", img_name))
				os.rename(os.path.join(rgb2_path, img_name), os.path.join(DATA_PATH, "positive", img_name))
			else:
				# save to non traversable folder
				os.rename(os.path.join(rgb1_path, img_name), os.path.join(DATA_PATH, "negaive", img_name))
				os.rename(os.path.join(rgb2_path, img_name), os.path.join(DATA_PATH, "negaive", img_name))



def maybe_make_dir(path, folder_name):
	"""
	If the desired folder does not exist, create it.
	"""
	try:
		os.mkdir(os.path.join(path, folder_name))
		print(f"Folder {folder_name} created")
	except FileExistsError:
		pass


