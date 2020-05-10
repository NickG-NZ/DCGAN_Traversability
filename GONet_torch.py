"""A pytorch implementation of GONet"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# torch
import torch


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


class Generator(torch.Module):
	"""Generates fake images of how the scene should
		appear if it is traversable"""
	def __init__(self):
		super().__init__(self)

