import torch


class Subsampler:
	
	def __init__(self, num_samples, positive_fraction, bg_label):
		"""
		Args:
			num_samples (int): The total number of labels with value >= 0 to return.
				Values that are not sampled will be filled with -1 (ignore).
			positive_fraction (float): The number of subsampled labels with values > 0
				is `min(num_positives, int(positive_fraction * num_samples))`. The number
				of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
				In order words, if there are not enough positives, the sample is filled with
				negatives. If there are also not enough negatives, then as many elements are
				sampled as is possible.
			bg_label (int): label index of background ("negative") class.
		"""
		self.num_samples = num_samples
		self.positive_fraction = positive_fraction
		self.bg_label = bg_label

	def __call__(self, labels):
		"""
		Return `num_samples` (or fewer, if not enough found)
		random samples from `labels` which is a mixture of positives & negatives.
		It will try to return as many positives as possible without
		exceeding `positive_fraction * num_samples`, and then try to
		fill the remaining slots with negatives.

		Args:
			labels (Tensor): (N, ) label vector with values:
				* -1: ignore
				* 0: background ("negative") class
				* otherwise: one or more foreground ("positive") classes
		Returns:
			pos_idx, neg_idx (Tensor):
				1D vector of indices. The total length of both is `num_samples` or fewer.
		"""
		positive = ((labels >= 0) & (labels != self.bg_label)).nonzero(as_tuple=True)[0]
		negative = (labels == self.bg_label).nonzero(as_tuple=True)[0]

		num_pos = int(self.num_samples * self.positive_fraction)
		# protect against not enough positive examples
		num_pos = min(len(positive), num_pos)
		num_neg = self.num_samples - num_pos
		# protect against not enough negative examples
		num_neg = min(len(negative), num_neg)

		# randomly select positive and negative examples
		pos_perm = torch.randperm(len(positive), device=positive.device)[:num_pos]
		neg_perm = torch.randperm(len(negative), device=negative.device)[:num_neg]

		return positive[pos_perm], negative[neg_perm]

	def return_as_mask(self, labels):
		pos_idx, neg_idx = self(labels)
		labels.fill_(-1)
		labels.scatter_(0, pos_idx, 1)
		labels.scatter_(0, neg_idx, 0)
		return labels