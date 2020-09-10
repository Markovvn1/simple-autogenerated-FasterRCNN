import torch
import torch.nn as nn

class Anchors(nn.Module):

	def __init__(self, sizes, ratios, stride):
		super().__init__()
		assert isinstance(ratios, (list, tuple))
		assert isinstance(sizes, (list, tuple))
		assert isinstance(stride, int)

		r, s = self.__meshgrid(torch.Tensor(ratios).sqrt(), torch.Tensor(sizes))
		shift = torch.full((len(r),), stride / 2)

		self.stride = stride
		self.register_buffer("_cell_anchors", torch.stack([shift, shift, s * r, s / r]))

	def __meshgrid(self, x, y):
		return x.repeat(len(y)), y.view(-1, 1).repeat(1, len(x)).view(-1)

	def __len__(self):
		return self._cell_anchors.size(1)

	def forward(self, feature_hw):
		assert len(feature_hw) == 2

		device = self._cell_anchors.device
		sy = torch.arange(0, feature_hw[0] * self.stride, self.stride, device=device)
		sx = torch.arange(0, feature_hw[1] * self.stride, self.stride, device=device)

		yxhw = self._cell_anchors[:, :, None, None].repeat(1, 1, len(sy), len(sx))
		yxhw[0] += sy[None, :, None]
		yxhw[1] += sx[None, None, :]
		return yxhw


class MultiAnchors(nn.Module):

	def __init__(self, sizes, ratios, strides):
		super().__init__()

		sizes = self._broadcast_params(sizes, len(strides), "sizes")
		ratios = self._broadcast_params(ratios, len(strides), "ratios")

		self.anchors = []

		for si, r, st in zip(sizes, ratios, strides):
			self.anchors.append(Anchors(si, r, st))
			self.add_module(f"anchors{st}", self.anchors[-1])

	def __len__(self):
		return sum([len(a) for a in self.anchors])

	def _broadcast_params(params, num_features, name):
		"""
		If one size (or aspect ratio) is specified and there are multiple feature
		maps, we "broadcast" anchors of that single size (or aspect ratio)
		over all feature maps.

		If params is list[float], or list[list[float]] with len(params) == 1, repeat
		it num_features time.

		Returns:
			list[list[float]]: param for each feature
		"""
		assert isinstance(params, (list, tuple)), f"{name} in anchor generator has to be a list! Got {params}."
		assert len(params), f"{name} in anchor generator cannot be empty!"

		if not isinstance(params[0], (list, tuple)): return [params] * num_features
		if len(params) == 1: return list(params) * num_features

		assert len(params) == num_features, (
			f"Got {name} of length {len(params)} in anchor generator, "
			f"but the number of input features is {num_features}!")
		return params

	def forward(self, features_hw):
		assert isinstance(features_hw, (list, tuple))
		assert len(features_hw) == len(self.anchors)
		return [anchors(hw) for hw, anchor in zip(features_hw, self.anchors)]
