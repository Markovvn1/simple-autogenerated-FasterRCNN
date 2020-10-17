from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"resnet.CNNMetrics", "resnet.BasicStem", "resnet.BasicBlock",
			"resnet.BottleneckBlock", "resnet.ResNetStage", "resnet.ResNet"})

	def _assert_cfg(self, module, cfg):
		if module in ["resnet.CNNMetrics", "resnet.ResNetStage"]:
			assert not cfg
			return
		if module in ["resnet.BasicStem", "resnet.BasicBlock", "resnet.BottleneckBlock"]:
			assert isinstance(cfg["use_norm"], bool)
			if module == "resnet.BottleneckBlock":
				assert isinstance(cfg["stride_in_1x1"], bool)
			return
		assert module == "resnet.ResNet"

		is_pos_int = lambda x: isinstance(x, int) and x > 0

		assert cfg["depth"] in [18, 34, 50, 101, 152]

		assert cfg["norm"] in ["None", "BN", "FrozenBN"]
		assert is_pos_int(cfg["num_groups"])
		assert is_pos_int(cfg["width_per_group"])
		assert is_pos_int(cfg["stem_out_channels"])
		assert is_pos_int(cfg["res2_out_channels"])

		if cfg["depth"] in [18, 34]:
			assert cfg["res2_out_channels"] == 64, "Must set res2_out_channels = 64 for R18/R34"
			assert cfg["res5_dilation"] == 1, "Must set res5_dilation = 1 for R18/R34"
			assert cfg["num_groups"] == 1, "Must set num_groups = 1 for R18/R34"
		else:
			assert cfg["res5_dilation"] in [1, 2]

		assert all([i in ["stem", "res2", "res3", "res4", "res5"] for i in cfg["out_features"]])

	def _dependencies(self, module, global_params, cfg):
		if module == "resnet.CNNMetrics": return {}
		if module == "resnet.ResNetStage": return {"resnet.CNNMetrics": None}

		if module in ["resnet.BasicStem", "resnet.BasicBlock", "resnet.BottleneckBlock"]:
			if global_params["mode"] == "train":
				return {"resnet.CNNMetrics": None, "freeze_batchnorm.ModuleWithFreeze": None,
					"conv_wrapper.Conv2d": None}
			else:
				return {"resnet.CNNMetrics": None}

		assert module == "resnet.ResNet"
		use_norm = {"use_norm": cfg["norm"] != "None"}
		res = {"backbone.Backbone": None, "resnet.ResNetStage": None, "resnet.BasicStem": use_norm}
		if cfg["depth"] in [18, 34]:
			res["resnet.BasicBlock"] = use_norm
		else:
			res["resnet.BottleneckBlock"] = {**use_norm, "stride_in_1x1": cfg["stride_in_1x1"]}
		
		if global_params["mode"] == "train":
			res["conv_wrapper.Conv2d"] = None
			if cfg["norm"] == "FrozenBN":
				res["freeze_batchnorm.FrozenBatchNorm2d"] = None

		return res

	def _init_file(self, dep):
		if "resnet.ResNet" in dep: return "from .resnet import ResNet"
		return None

	def _generate(self, global_params, dep, childs):
		res = []
		res.append("""\
import torch
import torch.nn as nn
import torch.nn.functional as F\n\n""")

		if "backbone.Backbone" in childs:
			res.append("from .backbone import Backbone\n""")

		layers = {"conv_wrapper.Conv2d", "freeze_batchnorm.FrozenBatchNorm2d", "freeze_batchnorm.ModuleWithFreeze"} & childs.keys()
		if layers:
			res.append("from ..layers import " + ", ".join([i[i.find(".")+1:] for i in layers]) + "\n")

		if "resnet.CNNMetrics" in dep:
			res.extend(_generate_CNNMetrics(global_params, dep["resnet.CNNMetrics"]))
		if "resnet.BasicStem" in dep:
			res.extend(_generate_BasicStem(global_params, dep["resnet.BasicStem"]))
		if "resnet.BasicBlock" in dep:
			res.extend(_generate_BasicBlock(global_params, dep["resnet.BasicBlock"]))
		if "resnet.BottleneckBlock" in dep:
			res.extend(_generate_BottleneckBlock(global_params, dep["resnet.BottleneckBlock"]))
		if "resnet.ResNetStage" in dep:
			res.extend(_generate_ResNetStage(global_params, dep["resnet.ResNetStage"]))
		if "resnet.ResNet" in dep:
			res.extend(_generate_ResNet(global_params, dep["resnet.ResNet"]))

		return "".join(res)


class _Conv2d:

	def __init__(self, mode, use_norm):
		self.mode = mode
		self.use_norm = use_norm

	def __call__(self, in_channels, out_channels, kernel_size, stride=None, padding=None,
		bias=None, groups=None, dilation=None, norm_cls=None, t=2):
		
		if self.mode == "test":
			norm_cls = None
			if self.use_norm:
				bias = None

		t = "\t" * (t+1)

		main_params = [str(in_channels), str(out_channels), str(kernel_size)]
		if stride is not None: main_params.append(f"stride={stride}")
		if padding is not None: main_params.append(f"padding={padding}")
		if bias is not None: main_params.append(f"bias={bias}")
		if groups is not None: main_params.append(f"groups={groups}")
		if dilation is not None: main_params.append(f"dilation={dilation}")
		params = []
		if norm_cls is not None: params.append(f"norm_cls={norm_cls}")
		if self.mode == "train": params.append(f"init=\"c2_msra_fill\"")
		
		res = []
		if len(main_params) <= 5:
			main_params = ", ".join(main_params)
		else:
			main_params = ", ".join(main_params[:3]) + ",\n" + t + ", ".join(main_params[3:])

		res.append(f"{'nn.' if self.mode == 'test' else ''}Conv2d({main_params}")
		if params:
			res.append(",\n" + t + ", ".join(params))
		res.append(")")

		return "".join(res)


def _generate_CNNMetrics(global_params, cfg):
		return ["""\n
class CNNMetrics:

	def __init__(self, in_channels, out_channels, stride):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride\n"""]


def _generate_BasicStem(global_params, cfg):
	mode = global_params["mode"]
	conv = _Conv2d(mode, cfg["use_norm"])

	return [f"""\n
class BasicStem(CNNMetrics{"" if mode == "test" else ", ModuleWithFreeze"}, nn.Module):

	def __init__(self, in_channels, out_channels{"" if mode == "test" else ", norm_cls"}):
		CNNMetrics.__init__(self, in_channels, out_channels, 4)
		nn.Module.__init__(self)

		self.conv1 = {conv("in_channels", "out_channels", 7, 2, 3, False, None, None, "norm_cls", t=2)}

	def forward(self, x):
		x = F.relu_(self.conv1(x))
		x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
		return x\n"""]


def _generate_BasicBlock(global_params, cfg):
	mode = global_params["mode"]
	conv = _Conv2d(mode, cfg["use_norm"])

	return [f"""\n
class BasicBlock(CNNMetrics{"" if mode == "test" else ", ModuleWithFreeze"}, nn.Module):

	def __init__(self, in_channels, out_channels, stride{"" if mode == "test" else ", norm_cls"}):
		CNNMetrics.__init__(self, in_channels, out_channels, stride)
		nn.Module.__init__(self)

		if in_channels == out_channels:
			self.shortcut = lambda x: x
		else:
			self.shortcut = {conv("in_channels", "out_channels", 1, "stride", None, False, None, None, "norm_cls", t=3)}
	
		self.conv1 = {conv("in_channels", "out_channels", 3, "stride", 1, False, None, None, "norm_cls", t=2)}
		self.conv2 = {conv("out_channels", "out_channels", 3, 1, 1, False, None, None, "norm_cls", t=2)}

	def forward(self, x):
		out = F.relu_(self.conv1(x))
		out = F.relu_(self.conv2(out) + self.shortcut(x))
		return out\n"""]


def _generate_BottleneckBlock(global_params, cfg):
	mode = global_params["mode"]
	conv = _Conv2d(mode, cfg["use_norm"])

	return [f"""\n
class BottleneckBlock(CNNMetrics{"" if mode == "test" else ", ModuleWithFreeze"}, nn.Module):

	def __init__(self, in_channels, out_channels, stride, bottleneck_channels,
		dilation{"" if mode == "test" else ", norm_cls"}, num_groups):
		CNNMetrics.__init__(self, in_channels, out_channels, stride)
		nn.Module.__init__(self)

		if in_channels == out_channels:
			self.shortcut = nn.Identity()
		else:
			self.shortcut = {conv("in_channels", "out_channels", 1, "stride", None, False, None, None, "norm_cls", t=3)}

		self.conv1 = {conv("in_channels", "bottleneck_channels", 1, "stride" if cfg["stride_in_1x1"] else None, None, False, None, None, "norm_cls", t=2)}
		self.conv2 = {conv("bottleneck_channels", "bottleneck_channels", 3, None if cfg["stride_in_1x1"] else "stride", "dilation", False, "num_groups", "dilation", "norm_cls", t=2)}
		self.conv3 = {conv("bottleneck_channels", "out_channels", 1, None, None, False, None, None, "norm_cls", t=2)}

	def forward(self, x):
		out = F.relu_(self.conv1(x))
		out = F.relu_(self.conv2(out))
		out = F.relu_(self.conv3(out) + self.shortcut(x))
		return out\n"""]


def _generate_ResNetStage(global_params, cfg):
	res = []
	res.append("""\n
class ResNetStage(CNNMetrics, nn.Sequential):

	def __init__(self, block_class, in_channels, out_channels,
		first_stride, num_blocks, **kwargs):
		CNNMetrics.__init__(self, in_channels, out_channels, first_stride)

		res = [block_class(in_channels, out_channels, first_stride, **kwargs)]
		for i in range(num_blocks-1):
			res.append(block_class(out_channels, out_channels, 1, **kwargs))

		nn.Sequential.__init__(self, *res)\n""")

	if global_params["mode"] == "train":
		res.append("""
	def freeze(self):
		for block in self.children():
			block.freeze()\n""")
	return res


def _generate_ResNet(global_params, cfg):
	mode = global_params["mode"]

	q = "\""
	norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[cfg["norm"]]

	res = []
	res.append(f"""\n
class ResNet(Backbone):

	\"\"\"
	Properties of class:
		stages_list (list[str]): Непустой список всех stage'ей. Отсортированный
			по порядку выполнения
	\"\"\"

	def __init__(self, in_channels, out_features=None):
		\"\"\"
		Args:
			in_channels (int): Количество каналов на входе
			out_features (list[str]): Непустой список stage'ей, которые должны
				быть возвращены после вызова forward. Допустимые элементы:
				"stem", "res2", "res3", "res4", "res5"
				Если None, то результатом будет последний слой
		\"\"\"
		stages_list = ["stem", "res2", "res3", "res4", "res5"]

		assert isinstance(in_channels, int) and in_channels > 0
		if out_features is None: out_features = ["res5"]
		assert isinstance(out_features, list) and len(out_features) != 0
		assert len(out_features) == len(set(out_features))
		for item in out_features:
			assert item in stages_list

		super().__init__()

		out_features = [i for i in stages_list if i in out_features] # sort
		stages_list = stages_list[:stages_list.index(out_features[-1])+1]

		self._build_net(in_channels, out_features, stages_list)
		out_channels, out_stride = self._extract_channels_and_stages()
		self._init(in_channels, out_features, out_channels, out_stride)

	def _build_net(self, in_channels, out_features, stages_list):
		self.stages = []

		self.stem = stage = BasicStem(in_channels, {cfg["stem_out_channels"]}{"" if mode == "test" else ", norm_cls="+norm})
		self.stages.append(("stem" in out_features, stage))
		if len(stages_list) == 1: return

""")
	num_blocks_per_stage = {
		18: [2, 2, 2, 2],
		34: [3, 4, 6, 3],
		50: [3, 4, 6, 3],
		101: [3, 4, 23, 3],
		152: [3, 8, 36, 3],
	}[cfg["depth"]]

	block_class = "BasicBlock" if cfg["depth"] in [18, 34] else "BottleneckBlock"

	res.append(f"\t\tnum_blocks_per_stage = {num_blocks_per_stage}\n")
	res.append(f"\t\tout_channels         = {cfg['res2_out_channels']}\n")
	if block_class == "BottleneckBlock":
		res.append(f"\t\tbottleneck_channels  = {cfg['num_groups'] * cfg['width_per_group']}\n")

	res.append(f"""
		for i, stage_name in enumerate(stages_list[1:]):
			p = 2**i
			stage = ResNetStage({block_class},
				in_channels=stage.out_channels,
				out_channels=out_channels*p,
				first_stride=1 if i == 0{f" or stage_name == {q}res5{q}" if cfg["res5_dilation"] == 2 else ""} else 2,
				num_blocks=num_blocks_per_stage[i]""")
	if block_class == "BottleneckBlock":
		res.append(f""",
				bottleneck_channels=bottleneck_channels*p,
				dilation={f"2 if stage_name == {q}res5{q} else 1" if cfg["res5_dilation"] == 2 else "1"},
				num_groups={cfg["num_groups"]}""")
	if mode == "train":
		res.append(f""",
				norm_cls={norm}""")

	res.append(f""")
			self.add_module(stage_name, stage) # register stages
			self.stages.append((stage_name in out_features, stage))

	def _extract_channels_and_stages(self):
		out_stride = []
		cur_stride = 1

		for is_result, stage in self.stages:
			cur_stride *= stage.stride
			if is_result: out_stride.append(cur_stride)

		return [i[1].out_channels for i in self.stages if i[0]], out_stride

	def forward(self, x):
		self.assert_input(x)

		res = []
		for is_result, stage in self.stages:
			x = stage(x)
			if is_result: res.append(x)

		return res\n""")

	if mode == "train":
		res.append("""
	def freeze(self, freeze_at):
		\"\"\"Заморозить первые 'freeze_at' stages.

		Например, если freeze_at = 1, то заморожен будет только "stem".
		\"\"\"
		assert freeze_at in range(0, len(self.stages)+1)
		for _, stage in self.stages[:freeze_at]: stage.freeze()
		return self

	def extract(self):
		\"\"\"Подготовить модель для тестирования.

		Заменить все Conv2d на nn.Conv2d, объединив их с BatchNorm
		\"\"\"
		return Conv2d.extract_all(self)\n""")
	return res