import torch


def concat_images(images, size_divisibility):
	if len(images) == 1: return images[0].unsqueeze(0), [images[0].shape[-2:]]

	image_sizes = [i.shape[-2:] for i in images]
	h_max = (max(i[0] for i in image_sizes) + size_divisibility - 1) // size_divisibility
	w_max = (max(i[1] for i in image_sizes) + size_divisibility - 1) // size_divisibility
	# TODO: попробовать как в detectron2
	images = [F.pad(item, (0, w_max - sz[1], 0, h_max - sz[0])) for i, sz in zip(images, image_sizes)]
	return torch.stack(images), image_sizes