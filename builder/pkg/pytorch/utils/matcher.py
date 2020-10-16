from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"matcher.Matcher"})

	def _assert_cfg(self, module, cfg):
		assert not cfg

	def _dependencies(self, module, global_params, cfg):
		if module == "matcher.Matcher": return {"pairwise_iou.pairwise_iou": None}
		return {}

	def _init_file(self, dep):
		res = ", ".join([i[i.find(".")+1:] for i in dep.keys()])
		return "from .matcher import " + res

	def _generate(self, global_params, dep, childs):
		return _generate(global_params, dep)


def _generate(global_params, dep):
	return """\
import torch
from .pairwise_iou import pairwise_iou


class Matcher:

	def __init__(self, bg_threshold, fg_threshold, allow_low_quality_matches=False):
		\"\"\"Класс позволяет анализировать pairwise_iou.

		Args:
			bg_threshold (float): все что меньше получит значение 0
			fg_threshold (float): все что больше либо равно получит значение 1
			остальное получит значение -1
			allow_low_quality_matches (bool): if True, produce additional matches
				for predictions with maximum match quality lower than fg_threshold.
				See set_low_quality_matches_ for more details.
		\"\"\"
		self.bg_threshold = bg_threshold
		self.fg_threshold = fg_threshold
		self.allow_low_quality_matches = allow_low_quality_matches

	def _forward(self, target, anchors):
		if target.numel() == 0 or anchors.numel() == 0:
			# When no gt boxes exist, all predicted boxes are background(0)
			return x.new_full((x.size(1),), 0, dtype=torch.int64), \\
				x.new_full((x.size(1),), 0, dtype=torch.int8)

		x = pairwise_iou(target, anchors)

		matched_vals, matches = x.max(dim=0)
		match_labels = matches.new_full(matches.size(), -1, dtype=torch.int8)

		match_labels[matched_vals < self.bg_threshold] = 0
		match_labels[matched_vals >= self.fg_threshold] = 1

		if self.allow_low_quality_matches:
			highest_quality_foreach_gt = x.max(dim=1).values
			pred_inds_with_highest_quality = (x == highest_quality_foreach_gt[:, None]).nonzero(as_tuple=True)[1]

			# If an anchor was labeled positive only due to a low-quality match
			# with gt_A, but it has larger overlap with gt_B, it's matched index
			# will still be gt_B. This follows the implementation in Detectron,
			# and is found to have no significant impact.
			match_labels[pred_inds_with_highest_quality] = 1

		return matches, match_labels

	def __call__(self, target, anchors):
		\"\"\"
		Args:
			target (Tensor): Тензор, элементы которого будут сопоставляться anchors, (A, 4)
			anchors (Tensor): Тензор, которому будут сопоставляться элементы из target, (B, 4)

		Results:
			matched_idxs (Tensor[int64]): Индекс сопоставленного target для каждого anchor, (B,)
			gt_labels_i (Tensor[int8]): Массив состоящий из (0, -1, 1) и задающий какие элементы
				foreground, какие background, а какие игнорировать
		\"\"\"
		# Matching is memory-expensive. But the result is small
		try:
			return self._forward(target, anchors)
		except RuntimeError as e:
			if "CUDA out of memory. " not in str(e): raise
			print("Note: runing pairwise_iou on CPU.")
			matched_idxs, gt_labels_i = self.anchor_matcher(target.cpu(), anchors.cpu())
			return matched_idxs.to(device=target.device), gt_labels_i.to(device=target.device)\n"""
