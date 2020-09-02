import zlib, base64

def build_model(cfg, engine="pytorch"):
	with open("model.py", "w") as f:
		f.write(generate_model(cfg))

	return set()


def generate_model(cfg, engine="pytorch"):
	assert isinstance(engine, str)
	assert isinstance(cfg["FPN"]["out_channels"], int) and cfg["FPN"]["out_channels"] > 0
	assert all([i in ["stem", "res2", "res3", "res4", "res5"] for i in cfg["RESNETS"]["out_features"]])

	if engine == "pytorch":
		return generate_model_pytorch(cfg)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_model_pytorch(cfg):
	cfg_str = base64.b64encode(zlib.compress(str(cfg).encode(), 9)).decode()
	cfg_str = [cfg_str[i:i+77] for i in range(0, len(cfg_str), 77)]
	assert zlib.decompress(base64.b64decode("".join(cfg_str).encode())).decode() == str(cfg)
	cfg_str = "\n# ".join(cfg_str)

	return f"""###  Automatically-generated file  ###

import torch
import torch.nn as nn

from .resnet import ResNet
from .fpn import FPN

# Configuration (base64, zlib):
# {cfg_str}

class Model(nn.Module):

	def __init__(self, in_channels, num_classes):
		bottom_up = ResNet(in_channels=in_channels, out_features={cfg["RESNETS"]["out_features"]})
		self.backbone = FPN(bottom_up, out_channels={cfg["FPN"]["out_channels"]})

	def forward(self, x):
		return self.backbone(x)
"""