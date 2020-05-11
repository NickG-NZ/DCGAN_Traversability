"""A pytorch implementation of GONet"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T


USE_GPU = True

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


"""Image constants"""
nz = 100

#center of picture
xc = 310
yc = 321

yoffset = 310 
xoffset = 310
xyoffset = 275
XYc = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]

# resize parameters
rsizex = 128
rsizey = 128

# zeros
outlist = np.zeros(15)


class Generator(nn.Module):
	"""Encodes images into a lower dimensional representation"""
	def __init__(self):
		super().__init__()


class invGen(nn.Module):
	"""Generates fake images of how the scene should
		appear if it is traversable"""


def main():
	print(f"Using device: {device}")


if __name__ == "__main__":
	main()