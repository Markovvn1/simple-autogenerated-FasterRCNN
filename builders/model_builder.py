import zlib, base64

def build_model(cfg, test_only=False, lib_prefix=".libs.", engine="pytorch"):
	with open("model.py", "w") as f:
		f.write(generate_model(cfg, test_only, lib_prefix))

	return set()


def generate_model(cfg, test_only=False, lib_prefix=".libs.", engine="pytorch"):
	assert isinstance(test_only, bool)
	assert isinstance(lib_prefix, str)
	assert isinstance(engine, str)

	assert isinstance(cfg["FPN"]["out_channels"], int) and cfg["FPN"]["out_channels"] > 0
	assert all([i in ["stem", "res2", "res3", "res4", "res5"] for i in cfg["RESNETS"]["out_features"]])

	if engine == "pytorch":
		return generate_model_pytorch(cfg, test_only, lib_prefix)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_model_pytorch(cfg, test_only, lib_prefix):
	res = []
	res.append("###  Automatically-generated file  ###\n\n")

	def generate_imports():
		res.append("""\
import torch
import torch.nn as nn

from .resnet import ResNet
from .fpn import FPN\n""")

		if not test_only:
			res.append(f"from {lib_prefix}conv_wrapper import Conv2d\n")

		res.append("\n")

	def generate_config():
		cfg_str = base64.b64encode(zlib.compress(str(cfg).encode(), 9)).decode()
		cfg_str = [cfg_str[i:i+77] for i in range(0, len(cfg_str), 77)]
		assert zlib.decompress(base64.b64decode("".join(cfg_str).encode())).decode() == str(cfg)
		cfg_str = "# " + "\n# ".join(cfg_str)

		res.append("# Configuration (base64, zlib):\n")
		res.append(cfg_str)
		res.append("\n\n")

	def generate_Model():
		res.append(f"""\
class Model(nn.Module):

	def __init__(self, in_channels, num_classes):
		super().__init__()
		bottom_up = ResNet(in_channels=in_channels, out_features={cfg["RESNETS"]["out_features"]})
		self.backbone = FPN(bottom_up, out_channels={cfg["FPN"]["out_channels"]})

	def forward(self, x):
		return self.backbone(x)\n""")

		if not test_only:
			res.append("""
	def extract(self):
		\"\"\"Подготовить модель для тестирования.

		Заменить все Conv2d на nn.Conv2d, объединив их с BatchNorm
		\"\"\"
		return Conv2d.extract_all(self)\n""")

	generate_imports()
	generate_config()
	generate_Model()

	return "".join(res)