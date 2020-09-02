import os

def build_backbone(engine="pytorch"):
	with open("backbone.py", "w") as f:
		f.write(generate_backbone())

	return set()


def generate_backbone(engine="pytorch"):
	if engine == "pytorch":
		return generate_backbone_pytorch()
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_backbone_pytorch():
	return """import torch.nn as nn

class Backbone(nn.Module):

	\"\"\"
	Properties of class:
		in_channels (int): Количество каналов на входе
		out_strides (list[int]): Произведение всех stride до данного stage
		out_channels (list[int]): Количество каналов на данном stage
		out_features (list[str]): Непустой список stage'ей, которые должны
			быть возвращены после вызова forward. Отсортированный по
			порядку выполнения
		size_divisibility (int): разрешение входного слоя должно делиться на это число
	\"\"\"

	def __init__(self):
		super().__init__()		

	def _init(self, in_channels, out_features, out_channels, out_strides, size_divisibility=1):
		assert len(out_features) == len(out_channels) and len(out_channels) == len(out_strides)

		self.in_channels = in_channels
		self.out_features = out_features
		self.out_channels = out_channels
		self.out_strides = out_strides
		self.size_divisibility = size_divisibility

	def assert_input(self, x):
		assert x.dim() == 4 and x.size(1) == self.in_channels,\\
			f"Input shape have to be (N, {self.in_channels}, H, W). Got {x.shape} instead!"
		assert x.size(2) % self.size_divisibility == 0 and x.size(3) % self.size_divisibility == 0,\\
			f"Размеры входных изображений должны делиться на self.size_divisibility"
"""