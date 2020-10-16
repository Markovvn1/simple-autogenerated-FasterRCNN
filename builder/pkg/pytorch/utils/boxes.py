from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"boxes.clamp_", "boxes.nonempty", "boxes.xywh2xyxy", "boxes.xyxy2xywh"})

	def _assert_cfg(self, module, cfg):
		assert not cfg

	def _dependencies(self, module, global_params, cfg):
		return {}

	def _init_file(self, dep):
		return "from . import boxes as Boxes"

	def _generate(self, global_params, dep, childs):
		return _generate(global_params, dep)


def _generate(global_params, dep):
	res = []
	res.append("import torch\n")


	if "boxes.clamp_" in dep:
		res.append("""\n
def clamp_(boxes, box_size):
	\"\"\"
		Clip (in place) the boxes by limiting x coordinates to the range [0, width]
		and y coordinates to the range [0, height].

		Args:
			box_size (height, width): The clipping box's size.
	\"\"\"
	assert torch.isfinite(boxes).all(), "Box tensor contains infinite or NaN!"
	boxes[..., 0::2].clamp_(min=0, max=box_size[1])
	boxes[..., 1::2].clamp_(min=0, max=box_size[0])\n""")


	if "boxes.nonempty" in dep:
		res.append("""\n
def nonempty(boxes, threshold: float=0.0):
	\"\"\"
	Find boxes that are non-empty.
	A box is considered empty, if either of its side is no larger than threshold.

	Returns:
		Tensor:
			a binary vector which represents whether each box is empty
			(False) or non-empty (True).
	\"\"\"
	widths_keep = (boxes[:, 2] - boxes[:, 0]) > threshold
	heights_keep = (boxes[:, 3] - boxes[:, 1]) > threshold
	return widths_keep & heights_keep\n""")


	if "boxes.xywh2xyxy" in dep:
		res.append("""\n
def xywh2xyxy(boxes):
	res = torch.empty_like(boxes)
	wh_half = boxes[..., 2:] / 2
	res[..., :2] = boxes[..., :2] - wh_half
	res[..., 2:] = boxes[..., :2] + wh_half
	return res\n""")


	if "boxes.xyxy2xywh" in dep:
		res.append("""\n
def xyxy2xywh(boxes):
	res = torch.empty_like(boxes)
	res[..., :2] = (boxes[..., 2:] + boxes[..., :2]) / 2
	res[..., 2:] = boxes[..., 2:] - boxes[..., :2]
	return res\n""")

	return "".join(res)