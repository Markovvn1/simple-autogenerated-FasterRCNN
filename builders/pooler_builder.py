import os

FOLDER = "layers"


def build_pooler(cfg, test_only=False, engine="pytorch"):
	os.makedirs(FOLDER, exist_ok=True)

	code, libs = generate_pooler(cfg, test_only, engine)

	with open(os.path.join(FOLDER, "pooler.py"), "w") as f:
		f.write(code)

	return libs


def generate_pooler(cfg, test_only=False, engine="pytorch"):
	"""
	Генерирует код для RPN используя параметры из cfg.

	Args:
		cfg (Dict[String, Any]): Список параметров конфигурации ResNet:
			TODO

		test_only (Bool): Нужно ли генерировать код только для тестирования, или он
			будет использоваться и для обучения

		engine (String): Движок для которого будет сгенерирован код. Возможные значения:
			"pytorch"
	"""

	assert isinstance(test_only, bool)
	assert isinstance(engine, str)

	assert cfg["type"] in ["RoIAlign", "RoIAlignV2", "RoIPool"]
	if isinstance(cfg["resolution"], int):
		assert cfg["resolution"] > 0
	else:
		assert isinstance(cfg["resolution"], (list, tuple)) and len(cfg["resolution"]) == 2
		assert all([isinstance(i, int) and i > 0 for i in cfg["resolution"]])
		
	assert isinstance(cfg["sampling_ratio"], int)

	if engine == "pytorch":
		return generate_pooler_pytorch(cfg, test_only)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_pooler_pytorch(cfg, test_only):
	res = []
	libs = set()
	res.append("###  Automatically-generated file  ###\n\n")

	def generate_imports():
		res.append("from math import log2, isclose\n")
		res.append("from sys import float_info\n\n")

		res.append("import torch\n")
		res.append("import torch.nn as nn\n")
		
		if cfg["type"] in ["RoIAlign", "RoIAlignV2"]:
			res.append("from torchvision.ops import RoIAlign\n")
		elif cfg["type"] == "RoIPool":
			res.append("from torchvision.ops import RoIPool\n")
		res.append("\n")


	def generate_RoIPooler():
		res.append("""
class RoIPooler(nn.Module):
	\"\"\"
	Region of interest feature map pooler that supports pooling from one or more
	feature maps.
	\"\"\"

	def __init__(self, output_size, strides, sampling_ratio=-1, canonical_box_size=224, canonical_level=4):
		\"\"\"
		Args:
			output_size (int, tuple[int] or list[int]): output size of the pooled region,
				e.g., 14 x 14. If tuple or list is given, the length must be 2.
			strides (list[float]): The strides for each feature map. Must be power of 2.
			sampling_ratio (int): The `sampling_ratio` parameter for the RoIAlign op.
			canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
				is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
				pre-training).
			canonical_level (int): The feature map level index from which a canonically-sized box
				should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
				i.e., a box of size 224x224 will be placed on the feature with stride=16.
				The box placement for all boxes will be determined from their sizes w.r.t
				canonical_box_size. For example, a box whose area is 4x that of a canonical box
				should be used to pool features from feature level ``canonical_level+1``.

				Note that the actual input feature maps given to this module may not have
				sufficiently many levels for the input boxes. If the boxes are too large or too
				small for the input feature maps, the closest level will be used.
		\"\"\"
		super().__init__()

		assert canonical_box_size > 0
		if isinstance(output_size, int): output_size = (output_size, output_size)
		assert isinstance(output_size, (tuple, list)) and len(output_size) == 2
		assert isinstance(output_size[0], int) and isinstance(output_size[1], int)

		for s in strides: assert isinstance(s, int)
		for a, b in zip(strides[:-1], strides[1:]):
			assert b == 2 * a, f"Strides {a} {b} are not log2 contiguous"

		self.output_size = output_size

		min_level, max_level = log2(strides[0]), log2(strides[-1])
		assert isclose(min_level, int(min_level)), "Featuremap stride is not power of 2!"
		min_level, max_level = int(min_level), int(max_level)
		assert 0 < min_level and min_level <= max_level

		# function which use size=sqrt(area) to calculate idx of needed feature map
		eps = float_info.epsilon
		canonical_level -= log2(canonical_box_size)  # for optimization
		self.size2lavel_idx = lambda size: (canonical_level + (size + eps).log2())\\
			.floor_().clamp_(min_level, max_level).long() - min_level\n""")

		if cfg["type"] in ["RoIAlign", "RoIAlignV2"]:
			res.append(f"""
		self.level_poolers = nn.ModuleList(
			RoIAlign(output_size, 1/s, sampling_ratio, aligned={cfg["type"] == "RoIAlignV2"}) for s in strides)\n""")
		elif cfg["type"] == "RoIPool":
			res.append("""
			self.level_poolers = nn.ModuleList(RoIPool(output_size, 1/s) for s in strides)\n""")

		res.append("""
	def forward(self, x, boxes):
		\"\"\"
		Args:
			x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
				used to construct this module.
			boxes (Tensor): Tensor of shape (M, 5), which represent boxes over all batch. There are
				5 columns: batch_idx, x0, y0, x1, y1

		Returns:
			Tensor:
				A tensor of shape (M, C, output_size, output_size) where M is the total number of
				boxes aggregated over all N batch images and C is the number of channels in `x`.
		\"\"\"
		assert isinstance(x, list)
		assert isinstance(boxes, torch.Tensor) and boxes.size(1) == 5
		assert len(x) == len(self.level_poolers)
		assert len(set(i.size(1) for i in x)) == 1, "number of channels have to be the same"

		if len(self.level_poolers) == 1:
			return self.level_poolers[0](x[0], boxes)

		# Calculate area and use it to calculate level_assignments
		level_assignments = self.size2lavel_idx((boxes[:, 3:5] - boxes[:, 1:3]).prod(dim=1).sqrt())

		res = torch.empty((len(boxes), x[0].shape[1], *self.output_size), dtype=x[0].dtype, device=x[0].device)

		for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
			inds = (level_assignments == level).nonzero().view(-1)
			res[inds] = pooler(x_level, boxes[inds])

		return res\n""")

	generate_imports()
	generate_RoIPooler()

	return "".join(res), libs
