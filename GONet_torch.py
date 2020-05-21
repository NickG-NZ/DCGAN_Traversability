"""
A pytorch implementation of GONet
@author Rongfei Lu
"""

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


USE_GPU = True

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# Image constants/Latent input vector size
nz = 100

# center of picture
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
	"""
	Generates fake images of how the scene should
	appear if it is traversable
	"""
	def __init__(self):
		super().__init__()
		self.l0z = nn.Linear(nz, 8*8*512)
		self.dc1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
		self.dc2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
		self.dc3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
		self.dc4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
		self.bn0l = nn.BatchNorm1d(8*8*512, eps=2e-05, momentum=0.1)
		self.bn0 = nn.BatchNorm2d(512, eps=2e-05, momentum=0.1)
		self.bn1 = nn.BatchNorm2d(256, eps=2e-05, momentum=0.1)
		self.bn2 = nn.BatchNorm2d(128, eps=2e-05, momentum=0.1)
		self.bn3 = nn.BatchNorm2d(64, eps=2e-05, momentum=0.1)
 
	def forward(self, z):
		h = torch.reshape(F.relu(self.bn0l(self.l0z(z))), (z.shape[0], 512, 8, 8))
		h = F.relu(self.bn1(self.dc1(h)))
		h = F.relu(self.bn2(self.dc2(h)))
		h = F.relu(self.bn3(self.dc3(h)))
		x = (self.dc4(h))
		return x

class InvGen(nn.Module):
	"""
	Encodes images into a lower dimensional representation (size nz)
	"""
	def __init__(self):
		super().__init__()
		self.c0 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
		self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
		self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
		self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
		self.l4l = nn.Linear(8*8*512, nz)
		self.bn0 = nn.BatchNorm2d(64, eps=2e-05, momentum=0.1)
		self.bn1 = nn.BatchNorm2d(128, eps=2e-05, momentum=0.1)
		self.bn2 = nn.BatchNorm2d(256, eps=2e-05, momentum=0.1)
		self.bn3 = nn.BatchNorm2d(512, eps=2e-05, momentum=0.1)
	   
	def forward(self, x):
		h = F.relu(self.c0(x))
		h = F.relu(self.bn1(self.c1(h)))
		h = F.relu(self.bn2(self.c2(h))) 
		h = F.relu(self.bn3(self.c3(h)))
		flat = h.view(h.shape[0], -1)
		l = self.l4l(flat)
		return l

class Discriminator(nn.Module):
	"""Classifies whether the image input is generated or not """
	def __init__(self, mode='train'):
		super().__init__()
		self.mode = mode
		self.c0 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
		self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
		self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
		self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
		# self.l4l = nn.Linear(8*8*512, 2)\
		self.l4l = nn.Linear(8*8*512, 1)
		self.bn0 = nn.BatchNorm2d(64, eps=2e-05, momentum=0.1)
		self.bn1 = nn.BatchNorm2d(128, eps=2e-05, momentum=0.1)
		self.bn2 = nn.BatchNorm2d(256, eps=2e-05, momentum=0.1)
		self.bn3 = nn.BatchNorm2d(512, eps=2e-05, momentum=0.1)
        
	def forward(self, x):
		h = F.elu(self.c0(x))
		h = F.elu(self.bn1(self.c1(h)))
		h = F.elu(self.bn2(self.c2(h)))
		h = F.elu(self.bn3(self.c3(h)))
		flat = h.view(h.shape[0], -1)
		l = self.l4l(flat)
		if self.mode == 'train':
			return l
		else:
			return h

class Classification(nn.Module):
	""" Classfication Module that consists of FC layers and LSTM to produce
	the final output of traversability """
	def __init__(self):
		super().__init__()
		self.l_img = nn.Linear(3*128*128, 10)
		self.l_dis = nn.Linear(512*8*8, 10)
		self.l_fdis = nn.Linear(512*8*8, 10)
		self.l_LSTM = nn.LSTM(30, 30)
		self.l_FL = nn.Linear(30, 1)
		self.bnfl = nn.BatchNorm2d(2048*7*7, eps=2e-05, momentum=0.1)

	def reset_state(self):
		self.l_LSTM.reset_state()

	def set_state(self):
		self.l_LSTM.set_state()
        
	def forward(self, img_error, dis_error, dis_output):
		h = torch.reshape(torch.abs(img_error), (img_error.shape[0], 3*128*128))
		h = self.l_img(h)
		g = torch.reshape(torch.abs(dis_error), (dis_error.shape[0], 512*8*8))
		g = self.l_dis(g)
		f = torch.reshape(dis_output, (dis_output.shape[0], 512*8*8))
		f = self.l_fdis(f)
		con = torch.cat((h,g,f), dim=1)
		ls = self.l_LSTM(con)
		ghf = torch.sigmoid(self.l_FL(ls))
		return ghf

def weights_init(m):
	"""Weight initialization for all Conv, BatchNorm and Linear layers"""
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm") != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find("Linear") != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
	# check current device in use
	print("Using device:", device)

	# initialize model
	gen = Generator().to(device)
	gen.apply(weights_init)
	invg = InvGen().to(device)
	invg.apply(weights_init)
	dis = Discriminator().to(device)
	dis.apply(weights_init)
	fl = Classification().to(device)
	fl.apply(weights_init)


if __name__ == "__main__":
	main()
