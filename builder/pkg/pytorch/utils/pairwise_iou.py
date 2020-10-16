from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"pairwise_iou.pairwise_iou"})

	def _assert_cfg(self, module, cfg):
		assert not cfg

	def _dependencies(self, module, global_params, cfg):
		return {}

	def _init_file(self, dep):
		res = ", ".join([i[i.find(".")+1:] for i in dep.keys()])
		return "from .pairwise_iou import " + res

	def _generate(self, global_params, dep, childs):
		return _generate(global_params, dep)


def _generate(global_params, dep):
	return """\
import torch


def pairwise_iou(boxes1, boxes2):
	\"\"\"
	Args:
		boxes1, boxes2 (Tensor): (xmin, ymin, xmax, ymax), (A, 4)

	Returns:
		Tensor: IoU, (A, B)
	\"\"\"
	assert boxes1.dim() == 2 and boxes1.size(1) == 4
	assert boxes2.dim() == 2 and boxes2.size(1) == 4
	assert boxes1.device == boxes2.device

	inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (A, B, 2)
	inter.clamp_(min=0)  # (A, B, 2)
	inter = inter.prod(dim=2)  # (A, B)

	union = (boxes1[:, 2:] - boxes1[:, :2]).prod(dim=-1, keepdim=True) + (boxes2[:, 2:] - boxes2[:, :2]).prod(dim=-1) - inter

	# handle empty boxes
	return torch.where(union > 0, inter / union, torch.zeros(1, dtype=inter.dtype, device=inter.device))\n"""