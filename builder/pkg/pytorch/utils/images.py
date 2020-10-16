from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"images.concat_images"})

	def _assert_cfg(self, module, cfg):
		assert not cfg

	def _dependencies(self, module, global_params, cfg):
		return {}

	def _init_file(self, dep):
		return "from . import images as Images"

	def _generate(self, global_params, dep, childs):
		return _generate(global_params, dep)


def _generate(global_params, dep):
	return """\
import torch
import torch.nn.functional as F


def concat_images(images, size_divisibility):
	image_sizes = [i.shape[-2:] for i in images]
	sd = size_divisibility
	h_max = (max(i[0] for i in image_sizes) + sd - 1) // sd * sd
	w_max = (max(i[1] for i in image_sizes) + sd - 1) // sd * sd
	# TODO: попробовать как в detectron2
	images = [F.pad(i, (0, w_max - sz[1], 0, h_max - sz[0])) for i, sz in zip(images, image_sizes)]
	return torch.stack(images), image_sizes\n"""