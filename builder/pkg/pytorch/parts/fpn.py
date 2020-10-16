from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"fpn.FPN"})

	def _assert_cfg(self, module, cfg):
		assert module == "fpn.FPN"

		assert cfg["norm"] in ["None", "BN", "FrozenBN"]
		assert cfg["fuse_type"] in ["sum", "avg"]
		assert isinstance(cfg["out_channels"], int) and cfg["out_channels"] > 0

	def _dependencies(self, module, global_params, cfg):
		assert module == "fpn.FPN"

		res = {"backbone.Backbone": None}
		if global_params["mode"] == "train":
			res["conv_wrapper.Conv2d"] = None
			if cfg["norm"] == "FrozenBN":
				res["freeze_batchnorm.FrozenBatchNorm2d"] = None

		return res

	def _init_file(self, dep):
		if "fpn.FPN" in dep: return "from .fpn import FPN\n"
		return None

	def _generate(self, global_params, dep, childs):
		res = []
		res.append("""\
from math import log2
		
import torch
import torch.nn as nn
import torch.nn.functional as F\n""")
		
		if "backbone.Backbone" in childs:
			res.append("\nfrom .backbone import Backbone\n")

		layers = {"conv_wrapper.Conv2d", "freeze_batchnorm.FrozenBatchNorm2d"} & childs.keys()
		if layers:
			res.append("from ..layers import " + ", ".join([i[i.find(".")+1:] for i in layers]) + "\n")

		if "fpn.FPN" in dep:
			res.extend(_generate_FPN(global_params, dep["fpn.FPN"]))

		return "".join(res)


def _generate_FPN(global_params, cfg):
	mode = global_params["mode"]
	norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[cfg["norm"]]

	res = []
	res.append("""\n
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

	temp = ""
	if mode == "train" and cfg["norm"] != "None":  # not use bias
		temp = ", bias=False"
	if mode == "train":
		temp = temp + f",\n\t\t\t\tnorm_cls={norm}, init=\"c2_xavier_fill\""

	res.append(f"""\
		self.tail_conv = []
		for idx, in_channels in zip(out_features_idx[:-1], bottom_up.out_channels):
			lateral_conv = {"nn." if mode == "test" else ""}Conv2d(in_channels, out_channels, 1{temp})
			self.add_module(f"fpn_lateral{{idx}}", lateral_conv)

			output_conv = {"nn." if mode == "test" else ""}Conv2d(out_channels, out_channels, 3, padding=1{temp})
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
	return res