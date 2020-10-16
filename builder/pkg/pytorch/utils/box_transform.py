from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"box_transform.BoxTransform"})

	def _assert_cfg(self, module, cfg):
		assert not cfg

	def _dependencies(self, module, global_params, cfg):
		return {}

	def _init_file(self, dep):
		res = ", ".join([i[i.find(".")+1:] for i in dep.keys()])
		return "from .box_transform import " + res

	def _generate(self, global_params, dep, childs):
		return _generate(global_params, dep)


def _generate(global_params, dep):
	return """\
import math

import torch
import torch.nn as nn


# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
SCALE_CLAMP = math.log(1000.0 / 16)


class BoxTransform:

	def __init__(self, weights):
		assert isinstance(weights, (list, tuple)) and len(weights) == 4
		assert all([isinstance(i, (float, int)) for i in weights])
		self.weights = torch.tensor(weights, dtype=torch.float32)

	def get_deltas(self, anchors, target_boxes):
		\"\"\"
		Args:
			anchors (Tensor): anchors в формате xywh, (..., 4)
			target_boxes (Tensor): коробки в формате xyxy, (..., 4)
		\"\"\"
		assert anchors.size(-1) == 4
		assert target_boxes.device == anchors.device

		if target_boxes.shape != anchors.shape:
			anchors = anchors.expand(target_boxes.shape)

		target_wh = target_boxes[..., 2:] - target_boxes[..., :2]
		target_cxcy = target_boxes[..., :2] + target_wh / 2

		res = torch.empty_like(target_boxes)
		res[..., :2] = (target_cxcy - anchors[..., :2]) / anchors[..., 2:]
		res[..., 2:] = (target_wh / anchors[..., 2:]).log()
		return res * self.weights.to(device=res.device)

	def apply_deltas(self, anchors, deltas):
		\"\"\"
		Args:
			anchors (Tensor): anchors в формате xywh, (..., 4)
			deltas (Tensor): transformation deltas, (..., 4)
		\"\"\"
		assert anchors.size(-1) == 4
		assert deltas.device == anchors.device

		if deltas.shape != anchors.shape:
			anchors = anchors.expand(deltas.shape)

		deltas = deltas / self.weights.to(device=deltas.device)
		pred_xy = deltas[..., :2] * anchors[..., 2:] + anchors[..., :2]
		pred_wh = deltas[..., 2:].clamp_max(SCALE_CLAMP).exp() * anchors[..., 2:]

		res = torch.empty_like(deltas)
		res[..., :2] = pred_xy - pred_wh / 2
		res[..., 2:] = pred_xy + pred_wh / 2
		return res\n"""
