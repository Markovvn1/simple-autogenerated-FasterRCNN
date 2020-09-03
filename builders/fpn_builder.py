def build_fpn(cfg, test_only=False, lib_prefix=".libs.", engine="pytorch"):
	code = generate_fpn(cfg, test_only, lib_prefix, engine)

	with open("fpn.py", "w") as f:
		f.write(code)

	if not test_only:
		return {"conv_wrapper.py"}
	else:
		return set()


def generate_fpn(cfg, test_only=False, lib_prefix=".libs.", engine="pytorch"):
	"""
	Генерирует код для FPN используя параметры из cfg.

	Args:
		cfg (Dict[String, Any]): Список параметров конфигурации FPN:
			"norm": String - Слой нормализации, который будет использоваться в сети
				Возможные значения: "None", "BN", "FrozenBN"
			"fuse_type": String - Способ объединения слоев.
				Возможные значения: "sum", "avg"

		test_only (Bool): Нужно ли генерировать код только для тестирования, или он
			будет использоваться и для обучения

		engine (String): Движок для которого будет сгенерирован код. Возможные значения:
			"pytorch"
	"""

	assert isinstance(test_only, bool)
	assert isinstance(lib_prefix, str)
	assert isinstance(engine, str)

	assert cfg["norm"] in ["None", "BN", "FrozenBN"]
	assert cfg["fuse_type"] in ["sum", "avg"]

	if engine == "pytorch":
		return generate_fpn_pytorch(cfg, test_only, lib_prefix)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_fpn_pytorch(cfg, test_only=False, lib_prefix=".libs."):
	res = []
	res.append("###  Automatically-generated file  ###\n\n")

	norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[cfg["norm"]]

	def generate_imports():
		res.append("from math import log2\n")
		res.append("\n")
		res.append("import torch\n")
		res.append("import torch.nn as nn\n")
		res.append("import torch.nn.functional as F\n")
		res.append("\n")
		res.append("from .backbone import Backbone\n")

		if not test_only:
			if cfg["norm"] == "FrozenBN":
				res.append(f"from {lib_prefix}freeze_batchnorm import FrozenBatchNorm2d\n")
			res.append(f"from {lib_prefix}conv_wrapper import Conv2d\n")

		res.append("\n\n")

	def generate_FPN():
		res.append("""\
class FPN(Backbone):

	def __init__(self, bottom_up, out_channels):
		assert isinstance(bottom_up, Backbone)
		for a, b in zip(bottom_up.out_strides[:-1], bottom_up.out_strides[1:]):
			assert b == 2 * a, f"Strides {a} {b} are not log2 contiguous"

		super().__init__()
		self.bottom_up = bottom_up

		out_strides = bottom_up.out_strides + [bottom_up.out_strides[-1] * 2]
		out_features_idx = [str(int(log2(i))) for i in out_strides]
		out_features = ["p"+i for i in out_features_idx]

		super()._init(bottom_up.in_channels, out_features,
			[out_channels] * len(out_features),  # out_channels
			out_strides, size_divisibility=bottom_up.out_strides[-1])\n\n""")

		use_bias = test_only or cfg["norm"] == "None"

		temp = f""",\n\t\t\t\tnorm_cls={norm}, init=\"c2_xavier_fill\""""

		res.append(f"""\
		self.tail_conv = []
		for idx, in_channels in zip(out_features_idx[:-1], bottom_up.out_channels):
			lateral_conv = {"nn." if test_only else ""}Conv2d(in_channels, out_channels,
				kernel_size=1{", bias=False" if not use_bias else ""}{"" if test_only else temp})
			self.add_module(f"fpn_lateral{{idx}}", lateral_conv)

			output_conv = {"nn." if test_only else ""}Conv2d(out_channels, out_channels,
				kernel_size=3, padding=1{", bias=False" if not use_bias else ""}{"" if test_only else temp})
			self.add_module(f"fpn_output{{idx}}", output_conv)

			self.tail_conv.append((lateral_conv, output_conv))

		self.tail_conv = self.tail_conv[::-1]\n\n""")

		res.append("""\
	def _concat(self, x, top_down):
		if top_down is None: return x
		""")
		if cfg["fuse_type"] == "sum":
			res.append("""return x + F.interpolate(top_down, scale_factor=2, mode="nearest")\n""")
		else:
			res.append("""return (x + F.interpolate(top_down, scale_factor=2, mode="nearest")) / 2\n""")

		res.append("""
	def forward(self, x):
		self.assert_input(x)

		x = self.bottom_up(x)[::-1]

		res = []
		prev_features = None
		for features, (lateral_conv, output_conv) in zip(x, self.tail_conv):
			prev_features = self._concat(lateral_conv(features), prev_features)
			res.append(output_conv(prev_features))

		res = res[::-1]
		# LastLevelMaxPool
		res.append(F.max_pool2d(res[-1], kernel_size=1, stride=2, padding=0))

		return res\n""")

	generate_imports()
	generate_FPN()

	return "".join(res)