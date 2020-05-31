"""
Utiltiies for parsing Raw [synced + rectified] Kitti data
Uses the velocity data to automatically classify image frames as traversable / non-traversable
"""

"""
1) Use velocity threshold to separate images into two folders
	- likely traversable/positive
	- likely untraversable/negative
2) Create corresponding data files which hold the velocities (and other measurements) for both folders
	
3) resize the images to 128x128

2) Check the images in the positive folder (along with their velocities) to ensure they seem reasonable

5) Split the main positive folder into train and validation folders (use ~ 80-20 split) in order
"""

import os
import numpy as np
import shutil
from PIL import Image

#********* Change this for your machines *****************
DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Kitti_Data"
RAW_DATA_FOLDERS = ["2011_09_26_drive_0057_sync"]

# Automatically label the data using a velocity threshold
V_THRESH = 1.4  # [m/s] (5 km/h)
vf_idx = 8  # index of forward velocity
vl_idx = 9  # index of lateral velocity


def auto_label_kitti(data_folder, prefix):
	"""
	Automatically label data based on velocity threshold
	Puts images into 2 folders "positive" and "negative"
	Puts IMU/GPS (oxts) into 2 folders "oxts_positive", "oxts_negative"

	inputs:
	- data_folder: (string) the folder name for the raw kitti data
	- prefix: (string) prefix to add to the front of all imgs that are moved
				from the given Kitti data folder. All Kitti raw data-sets use the same filenames
				so it is neccessary to differentiate the files when concatenating multiple data-sets
	"""
	maybe_make_dir(DATA_PATH, "positive")
	maybe_make_dir(DATA_PATH, "negative")
	maybe_make_dir(DATA_PATH, "oxts_positive")
	maybe_make_dir(DATA_PATH, "oxts_negative")

	oxts_path = os.path.join(DATA_PATH, data_folder, "oxts", "data")       # auxiliary data (IMU/GPS)
	rgb1_path = os.path.join(DATA_PATH, data_folder, "image_02", "data")  # first RGB camera
	rgb2_path = os.path.join(DATA_PATH, data_folder, "image_03", "data")  # second RGB camera

	oxts = sorted(os.listdir(oxts_path))
	rgb1_img_list = sorted(os.listdir(rgb1_path))
	rgb2_img_list = sorted(os.listdir(rgb2_path))
	n_files = len(oxts)

	print(f"Starting auto-labelling for folder {data_folder}")
	print(f"Number of files: {n_files}")
	for i in range(n_files):  # loop over all files and sort them using velocity
		file_id = oxts[i].split(".")[0]
		oxtdata = np.genfromtxt(os.path.join(oxts_path, oxts[i]))
		velocity = np.linalg.norm(np.array([oxtdata[vf_idx], oxtdata[vl_idx]])).item()

		img_name = file_id + ".png"
		imgs = [rgb1_img_list[i], rgb2_img_list[i]]
		for l, im in enumerate(imgs):  # check imgs with correct name exist
			assert im == img_name, \
			        f"couldn't find matching image {l + 1} for id:{file_id}"

		rgb_name_1 = prefix + "2_" + img_name
		rgb_name_2 = prefix + "3_" + img_name
		oxt_name = prefix + file_id + ".txt"
		if velocity > V_THRESH:
			split = "positive"
		else:
			split = "negative"
		os.rename(os.path.join(rgb1_path, img_name), os.path.join(DATA_PATH, split, rgb_name_1))
		os.rename(os.path.join(rgb2_path, img_name), os.path.join(DATA_PATH, split, rgb_name_2))
		shutil.copy2(os.path.join(oxts_path, file_id + ".txt"), os.path.join(DATA_PATH, "oxts_" + split, oxt_name))


def maybe_make_dir(path, folder_name):
	"""
	If the desired folder does not exist, create it.
	"""
	try:
		os.mkdir(os.path.join(path, folder_name))
		print(f"Folder {folder_name} created")
	except FileExistsError:
		pass


def resize_images():
	"""
	Crop and resize the sorted images to the 128x128 needed
	for the DCGAN
	"""
	print("Resizing images")
	folders = ["positive", "negative"]
	for folder in folders:
		path = os.path.join(DATA_PATH, folder)
		images = os.listdir(path)
		for i, img_name in enumerate(images):
			file_path = os.path.join(path, img_name)
			img = Image.open(file_path)
			(width, height) = img.size

			if width == 128:
				# image has already been processed
				continue
			left = (width - height) // 2
			right = left + height
			im_cropped = img.crop((left, 0, right, height))
			im_scaled = im_cropped.resize((128, 128))
			im_scaled.save(file_path)


def train_val_test_split():
	"""
	splits the data from the positive and negative
	examples into different training sets

	DCGAN / InvGen:
	================
	postiive - train (80 %), validation (20 %)

	Classifier
	===============
	mixed (800 images) - train(70 %), validation(20 %), test (10 %)
	"""
	# Take 400 positive and 400 negative examples for classifier
	maybe_make_dir(os.path.join(DATA_PATH, "data_train_annotated"), "positive")
	maybe_make_dir(os.path.join(DATA_PATH, "data_train_annotated"), "negative")
	maybe_make_dir(os.path.join(DATA_PATH, "data_vali_annotated"), "negative")





def _sort_and_resize(prefixes):

	"""Sort the data into positive/negative"""
	for i, data_folder in enumerate(RAW_DATA_FOLDERS):
		auto_label_kitti(data_folder, prefixes[i])

	"""resize the images"""
	resize_images()

	"""split data into train, validation, test folders"""


def main():
	# TODO: You need to fill in the RAW_DATA_FOLDERS list with the names of the Kitti raw data folders
	# TODO: You need to fill in the prefixes (these are the numbers at the very end of the downloaded folders names (eg. 57))
	# TODO: The number of prefixes should equal the number of raw data folders in the list RAW_DATA_FOLDERS
	prefixes = ["57", ""]

	# get rid of the following assert code, I just put this check here to make sure you read the comment above
	assert len(prefixes) > 2, "Nick made this error: You need to set the prefixes in the _sort_and_resize() function"

	"""Perform the actual sorting process"""
	_sort_and_resize(prefixes)



if __name__ == "__main__":
	main()


# Shouldn't need this hopefully
def undo_auto_label():
	"""
	undo the auto-label operation
	"""
	# move images out of here
	pos_path = os.path.join(DATA_PATH, "positive")
	neg_path = os.path.join(DATA_PATH, "negative")
	paths = [pos_path, neg_path]

	pos_imgs = os.listdir(pos_path)
	neg_imgs = os.listdir(neg_path)
	imgs = [pos_imgs, neg_imgs]

	n_pos = len(pos_imgs)
	n_neg = len(neg_imgs)
	lengths = [n_pos, n_neg]

	for i in range(2):
		for j in range(lengths[i]):
			img = imgs[i][j]
			kitti_raw_folder_idx = int(img[0]) - 1
			data_folder = RAW_DATA_FOLDERS[kitti_raw_folder_idx]

			rgb_num = img.split("_")[1]
			rgb_folder  = "image_0" + rgb_num
			original_im_name = "".join(img.split("_")[1])
			move_to_path = os.path.join(DATA_PATH, data_folder, rgb_folder, "data", original_im_name)  # correct RGB camera folder
			os.rename(os.path.join(paths[i], img), move_to_path)