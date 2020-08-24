###  Automatically-generated file  ###

# cfg = {'depth': 50, 'norm': 'BN', 'num_groups': 1, 'width_per_group': 64, 'stem_out_channels': 64, 'res2_out_channels': 256, 'stride_in_1x1': True, 'res5_dilation': 1}
# test_only = True, lib_prefix = "libs."

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMetrics:

	def __init__(self, in_channels, out_channels, stride):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride


class BasicStem(CNNMetrics, nn.Module):

	def __init__(self, in_channels, out_channels):
		CNNMetrics.__init__(self, in_channels, out_channels, 4)
		nn.Module.__init__(self)

		self.conv1 = nn.Conv2d(in_channels, out_channels,
			kernel_size=7, stride=2, padding=3)

	def forward(self, x):
		x = F.relu_(self.conv1(x))
		x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
		return x


class BottleneckBlock(CNNMetrics, nn.Module):

	def __init__(self, in_channels, out_channels, stride,
		bottleneck_channels, dilation):
		CNNMetrics.__init__(self, in_channels, out_channels, stride)
		nn.Module.__init__(self)

		if in_channels == out_channels:
			self.shortcut = lambda x: x
		else:
			self.shortcut = nn.Conv2d(in_channels, out_channels,
				kernel_size=1, stride=stride)

		self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
			kernel_size=1, stride=stride)
		self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
			kernel_size=3, padding=dilation, groups=1, dilation=dilation)
		self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
			kernel_size=1)

	def forward(self, x):
		out = F.relu_(self.conv1(x))
		out = F.relu_(self.conv2(out))
		out = F.relu_(self.conv3(out) + self.shortcut(x))
		return out


class ResNetStage(CNNMetrics, nn.Sequential):

	def __init__(self, block_class, in_channels, out_channels,
		first_stride, num_blocks, **kwargs):
		CNNMetrics.__init__(self, in_channels, out_channels, first_stride)

		res = [block_class(in_channels, out_channels, first_stride, **kwargs)]
		for i in range(num_blocks-1):
			res.append(block_class(out_channels, out_channels, 1, **kwargs))

		nn.Sequential.__init__(self, *res)


class ResNet(nn.Module):

	"""
	Properties of class:
		stride (dict[str, int]): Произведение всех stride до данного stage
		in_channels (int): Количество каналов на входе
		out_channels (dict[str, int]): Количество каналов на данном stage
		stages_list (list[str]): Непустой список всех stage'ей. Отсортированный
			по порядку выполнения
		out_features (list[str]): Непустой список stage'ей, которые должны
			быть возвращены после вызова forward. Отсортированный по
			порядку выполнения
	"""

	def __init__(self, in_channels, out_features=None):
		"""
		Args:
			in_channels (int): Количество каналов на входе
			out_features (list[str]): Непустой список stage'ей, которые должны
				быть возвращены после вызова forward. Допустимые элементы:
				"stem", "res2", "res3", "res4", "res5"
				Если None, то результатом будет последний слой
		"""
		stages_list = ["stem", "res2", "res3", "res4", "res5"]

		assert isinstance(in_channels, int) and in_channels > 0
		if out_features is None: out_features = ["res5"]
		assert isinstance(out_features, list) and len(out_features) != 0
		assert len(out_features) == len(set(out_features))
		for item in out_features:
			assert item in stages_list

		super().__init__()

		self.in_channels = in_channels
		self.out_features = [i for i in stages_list if i in out_features] # sort
		self.stages_list = stages_list[:stages_list.index(self.out_features[-1])+1]

		self._register_stages(self._build_net(in_channels, self.stages_list))

	def _build_net(self, in_channels, stages_list):
		res = []

		stage = BasicStem(in_channels, 64)
		res.append(("stem", "stem" in self.out_features, stage))
		if len(stages_list) == 1: return res

		num_blocks_per_stage = [3, 4, 6, 3]
		out_channels         = 256
		bottleneck_channels  = 64

		for i, stage_name in enumerate(stages_list[1:]):
			p = 2**i
			stage = ResNetStage(BottleneckBlock,
				in_channels=stage.out_channels,
				out_channels=out_channels*p,
				first_stride=1 if i == 0 else 2,
				num_blocks=num_blocks_per_stage[i],
				bottleneck_channels=bottleneck_channels*p,
				dilation=1)
			res.append((stage_name, stage_name in self.out_features, stage))

		return res

	def _register_stages(self, stages):
		self.stages = stages
		self.stride = {}
		self.out_channels = {}
		cur_stride = 1

		for name, _, stage in stages:
			self.add_module(name, stage) # register stages

			cur_stride *= stage.stride # Расчет stride и out_channels
			self.stride[name] = cur_stride
			self.out_channels[name] = stage.out_channels

	@torch.no_grad()
	def forward(self, x):
		assert x.dim() == 4 and x.size(1) == self.stem.in_channels,\
			f"ResNet takes an input of shape (N, {self.stem.in_channels}, H, W). Got {x.shape} instead!"

		res = {}
		for name, is_result, stage in self.stages:
			x = stage(x)
			if is_result: res[name] = x

		return res
