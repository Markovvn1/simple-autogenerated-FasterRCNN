import torch

def clamp_(boxes, box_size):
	"""
		Clip (in place) the boxes by limiting x coordinates to the range [0, width]
		and y coordinates to the range [0, height].

		Args:
			box_size (height, width): The clipping box's size.
	"""
	assert torch.isfinite(boxes).all(), "Box tensor contains infinite or NaN!"
	boxes[:, 0::2].clamp_(min=0, max=box_size[0])
	boxes[:, 1::2].clamp_(min=0, max=box_size[1])

def nonempty(boxes, threshold: float=0.0):
	"""
	Find boxes that are non-empty.
	A box is considered empty, if either of its side is no larger than threshold.

	Returns:
		Tensor:
			a binary vector which represents whether each box is empty
			(False) or non-empty (True).
	"""
	widths_keep = (boxes[:, 2] - boxes[:, 0]) > threshold
	heights_keep = (boxes[:, 3] - boxes[:, 1]) > threshold
	return widths_keep & heights_keep
