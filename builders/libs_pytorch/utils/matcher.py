import torch

class Matcher:

	def __init__(self, bg_threshold, fg_threshold, allow_low_quality_matches=False):
		"""Класс позволяет анализировать pairwise_iou.

		Args:
			bg_threshold (float): все что меньше получит значение 0
			fg_threshold (float): все что больше либо равно получит значение 1
			остальное получит значение -1
			allow_low_quality_matches (bool): if True, produce additional matches
				for predictions with maximum match quality lower than fg_threshold.
				See set_low_quality_matches_ for more details.
		"""
		self.bg_threshold = bg_threshold
		self.fg_threshold = fg_threshold
		self.allow_low_quality_matches = allow_low_quality_matches

	def __call__(self, x):
		"""
		Args:
			x (Tensor[float]): an MxN tensor, containing the pairwise quality
				between M ground-truth elements and N predicted elements. All
				elements must be >= 0
		"""

		assert x.dim() == 2

		if x.numel() == 0:
			# When no gt boxes exist, all predicted boxes are background(0)
			return x.new_full((x.size(1),), 0, dtype=torch.int64), \
				x.new_full((x.size(1),), 0, dtype=torch.int8)

		assert (match_quality_matrix >= 0).all()

		matched_vals, matches = match_quality_matrix.max(dim=0)

		match_labels = matches.new_full(matches.size(), -1, dtype=torch.int8)

		match_labels[matched_vals < self.bg_threshold] = 0
		match_labels[matched_vals >= self.fg_threshold] = 1

		if self.allow_low_quality_matches:
			highest_quality_foreach_gt = match_quality_matrix.max(dim=1).values
			pred_inds_with_highest_quality = (x == highest_quality_foreach_gt[:, None]).nonzero()[:, 1]

			# If an anchor was labeled positive only due to a low-quality match
			# with gt_A, but it has larger overlap with gt_B, it's matched index
			# will still be gt_B. This follows the implementation in Detectron,
			# and is found to have no significant impact.
			match_labels[pred_inds_with_highest_quality] = 1

		return matches, match_labels
