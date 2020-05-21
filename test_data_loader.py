
from GO_DataSet import GONetDataSet, display_num_images
import torchvision.transforms as T
from utils import Normalize
import os

DATA_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/GO_Data"
SAVE_PATH = "/home/nick/Documents/Conv_NN_CS231n/Project/DCGAN_Traversability/Training_Checkpoints/DCGAN"

transform = T.Compose([
	T.RandomHorizontalFlip(p=0.5),  # Flip image horizontally with p % chance
	T.ToTensor(),
	Normalize()])  # Convert images to range [-1, 1]


train = GONetDataSet(DATA_PATH, "train", transform=transform)
val = GONetDataSet(DATA_PATH, "vali", transform=transform)
test = GONetDataSet(DATA_PATH, "test", transform=transform)

datasets = [train, val, test]
display_num_images(datasets)

print("\n")
folders = ["positive_L", "positive_R"]
total_ims = {}
for set in datasets:
	print(set.split_name)
	total_ims[set.split_name] = 0
	for folder in folders:
		num_ims = len(os.listdir(os.path.join(DATA_PATH, set.split_name, folder)))
		print(folder, ": ", num_ims)
		total_ims[set.split_name] += num_ims
	print("total_ims: ", total_ims[set.split_name], "\n")

print(len(train))
print(len(val))
print(len(test))

print("\n")
print(train[len(train)])