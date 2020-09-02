###  Automatically-generated file  ###

import torch
import torch.nn as nn

from .resnet import ResNet
from .fpn import FPN

# Configuration
# sdfsdfsdfsdfsdf
# sdfsdfsdf
# sdfsdfsdfsdfsdfsdfsdf

class Model(nn.Module):

	def __init__(self, in_channels, num_classes):
		bottom_up = ResNet(in_channels=in_channels, out_features=["res2", "res3", "res4", "res5"])
		self.backbone = FPN(bottom_up, out_channels=256)

	def forward(self, x):
		return self.backbone(x)