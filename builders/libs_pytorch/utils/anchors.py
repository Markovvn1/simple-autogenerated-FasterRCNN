import math

import torch
import torch.nn as nn

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
SCALE_CLAMP = math.log(1000.0 / 16)

class AnchorsGrid:

	def __init__(self, grid):
		self.xywh = grid  # (A, 4)
		self._xy = grid[:, :2]
		self._wh = grid[:, 2:]

	def get_xyxy(self):
		res = torch.empty_like(self.xywh)
		wh_half = self._wh / 2
		res[:, :2] = self._xy - wh_half
		res[:, 2:] = self._xy + wh_half
		return res

	def get_xywh(self):
		return self.xywh

	def get_deltas(self, target_boxes):
		"""
		Args:
			target_boxes (Tensor[float]): коробки в формате xyxy, (A, 4)
		"""
		assert target_boxes.device == self.xywh.device
		assert target_boxes.dim() == 2 and len(target_boxes) == len(self.xywh)

		target_wh = target_boxes[:, 2:] - target_boxes[:, :2]
		target_cxcy = target_boxes[:, :2] + target_wh / 2

		res = torch.empty_like(target_boxes)
		res[:, :2] = (target_cxcy - self._xy) / self._wh
		res[:, 2:] = (target_wh / self._wh).log()
		return res

	def apply_deltas(self, deltas):
		"""
		Args:
			deltas (Tensor): transformation deltas of shape (N, 4)
		"""
		assert deltas.device == self.xywh.device
		assert deltas.dim() == 2 and len(deltas) == len(self.xywh)

		pred_xy = deltas[:, :2] * self._wh + self._xy
		pred_wh = torch.exp(deltas[:, 2:].clamp_max(SCALE_CLAMP)) * self._wh

		res = torch.empty_like(deltas)
		res[:, :2] = pred_xy - pred_wh / 2
		res[:, 2:] = pred_xy + pred_wh / 2
		return res


class Anchors(nn.Module):

	def __init__(self, sizes, ratios, stride):
		super().__init__()
		assert isinstance(ratios, (list, tuple))
		assert isinstance(sizes, (list, tuple))
		assert isinstance(stride, int)

		r, s = self.__meshgrid(torch.Tensor(ratios).sqrt(), torch.Tensor(sizes))
		shift = torch.full((len(r),), stride / 2)

		self.stride = stride
		self.register_buffer("_cell_anchors", torch.stack([shift, shift, s / r, s * r], dim=1))

	def __meshgrid(self, x, y):
		return x.repeat(len(y)), y.view(-1, 1).repeat(1, len(x)).view(-1)

	def __len__(self):
		return self._cell_anchors.size(1)

	def forward(self, feature_hw):
		assert len(feature_hw) == 2

		device = self._cell_anchors.device
		sy = torch.arange(0, feature_hw[0] * self.stride, self.stride, device=device)
		sx = torch.arange(0, feature_hw[1] * self.stride, self.stride, device=device)

		xywh = self._cell_anchors[None, None, :, :].repeat(len(sy), len(sx), 1, 1)
		xywh[..., 0] += sx[None, :, None]
		xywh[..., 1] += sy[:, None, None]
		return AnchorsGrid(xywh.flatten(end_dim=2))


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
