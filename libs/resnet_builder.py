
def generate_resnet(cfg, test_only=False, lib_prefix="", engine="pytorch"):
	"""
	Генерирует код для ResNet используя параметры из cfg.

	Args:
		cfg (Dict[String, Any]): Список параметров конфигурации ResNet:
			"depth": Int - Глубина сети. Возможные значения: 18, 34, 50, 101, 152
			"norm": String - Слой нормализации, который будет использоваться в сети
				Возможные значения: "None", "BN", "FrozenBN"
			"num_groups": Int - Количество групп для сверточных 3x3 слоев
			"width_per_group": Int - Количество каналов к каждой группе
			"stem_out_channels": Int - Количество каналов на выходе первого слоя
			"res2_out_channels": Int - Количество каналов на выходе второго слоя
			"stride_in_1x1": Bool - Будет ли stride проходить в слое 1x1 или же в слое 3x3
			"res5_dilation": Int - dilation в последнем слое. Возможные значения: 1, 2

		test_only (Bool): Нужно ли генерировать код только для тестирования, или он
			будет использоваться и для обучения

		engine (String): Движок для которого будет сгенерирован код. Возможные значения:
			"pytorch"
	"""

	assert isinstance(test_only, bool)
	assert isinstance(lib_prefix, str)
	assert isinstance(engine, str)

	is_pos_int = lambda x: isinstance(x, int) and x > 0

	assert cfg["depth"] in [18, 34, 50, 101, 152]

	assert cfg["norm"] in ["None", "BN", "FrozenBN"]
	assert is_pos_int(cfg["num_groups"])
	assert is_pos_int(cfg["width_per_group"])
	assert is_pos_int(cfg["stem_out_channels"])
	assert is_pos_int(cfg["res2_out_channels"])
	assert isinstance(cfg["stride_in_1x1"], bool)

	if cfg["depth"] in [18, 34]:
		assert cfg["res2_out_channels"] == 64, "Must set res2_out_channels = 64 for R18/R34"
		assert cfg["res5_dilation"] == 1, "Must set res5_dilation = 1 for R18/R34"
		assert cfg["num_groups"] == 1, "Must set num_groups = 1 for R18/R34"
	else:
		assert cfg["res5_dilation"] in [1, 2]

	if engine == "pytorch":
		return generate_resnet_pytorch(cfg, test_only, lib_prefix)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_resnet_pytorch(cfg, test_only, lib_prefix):
	res = []
	res.append("###  Automatically-generated file  ###\n\n")
	res.append(f"# cfg = {cfg}\n")
	res.append(f"# test_only = {test_only}, lib_prefix = \"{lib_prefix}\"\n\n")

	t = "\t"
	t3 = "\t"*3
	t2 = "\t"*2
	n = "\n"
	q = "\""
	norm = {"None": "NoBatchNorm", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[cfg["norm"]]

	def generate_imports():
		res.append("import torch\n")
		res.append("import torch.nn as nn\n")
		res.append("import torch.nn.functional as F\n")

		if not test_only:
			res.append("import fvcore.nn.weight_init as weight_init\n")

			res.append(f"\nfrom {lib_prefix}freeze_batchnorm import FrozenBatchNorm2d\n")
			res.append(f"from {lib_prefix}conv_wrapper import Conv2d\n")

		res.append("\n\n")


	def generate_CNNMetrics():
		res.append("""\
class CNNMetrics:

	def __init__(self, in_channels, out_channels, stride):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride\n\n\n""")


	def generate_ModuleWithFreeze():
		if not test_only:
			res.append("""\
class ModuleWithFreeze:

	def freeze(self):
		\"\"\"Freeze this module and return it.\"\"\"

		for p in self.parameters():
			p.requires_grad = False

		return FrozenBatchNorm2d.freeze_all_batchnorms(self)\n\n\n""")


	def Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, groups, dilation, norm=None, t=2):
		if test_only:
			norm = None
			if cfg["norm"] != "None":
				bias = None

		t = "\t" * (t+1)
		res = []
		res.append(f"{'nn.' if test_only else ''}Conv2d({in_channels}, {out_channels},\n")
		res.append(f"{t}kernel_size={kernel_size}")
		if stride is not None: res.append(f", stride={stride}")
		if padding is not None: res.append(f", padding={padding}")
		if bias is not None: res.append(f", bias={bias}")
		if groups is not None: res.append(f", groups={groups}")
		if dilation is not None: res.append(f", dilation={dilation}")
		res.append(f",\n{t}norm={norm})" if norm is not None else ")")
		return "".join(res)


	def generate_BasicStem():
		res.append(f"""\
class BasicStem(CNNMetrics{"" if test_only else ", ModuleWithFreeze"}, nn.Module):

	def __init__(self, in_channels, out_channels{"" if test_only else ", norm_cls"}):
		CNNMetrics.__init__(self, in_channels, out_channels, 4)
		nn.Module.__init__(self)

		self.conv1 = {Conv2d("in_channels", "out_channels", 7, 2, 3, False, None, None, "norm_cls(out_channels)", t=2)}
""")

		if not test_only:
			res.append("\t\tweight_init.c2_msra_fill(self.conv1)\n")

		res.append("""
	def forward(self, x):
		x = F.relu_(self.conv1(x))
		x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
		return x\n\n\n""")


	def generate_BasicBlock():
		res.append(f"""\
class BasicBlock(CNNMetrics{"" if test_only else ", ModuleWithFreeze"}, nn.Module):

	def __init__(self, in_channels, out_channels, stride{"" if test_only else ", norm_cls"}):
		CNNMetrics.__init__(self, in_channels, out_channels, stride)
		nn.Module.__init__(self)

		if in_channels == out_channels:
			self.shortcut = lambda x: x
		else:
			self.shortcut = {Conv2d("in_channels", "out_channels", 1, "stride", None, False, None, None, "norm_cls(out_channels)", t=3)}
{"" if test_only else t3+"weight_init.c2_msra_fill(self.shortcut)"+n}\
	
		self.conv1 = {Conv2d("in_channels", "out_channels", 3, "stride", 1, False, None, None, "norm_cls(out_channels)", t=2)}
{"" if test_only else t2+"weight_init.c2_msra_fill(self.conv1)"+n}\

		self.conv2 = {Conv2d("out_channels", "out_channels", 3, 1, 1, False, None, None, "norm_cls(out_channels)", t=2)}
{"" if test_only else t2+"weight_init.c2_msra_fill(self.conv2)"+n}\

	def forward(self, x):
		out = F.relu_(self.conv1(x))
		out = F.relu_(self.conv2(out) + self.shortcut(x))
		return out\n\n\n""")

	def generate_BottleneckBlock():
		res.append(f"""\
class BottleneckBlock(CNNMetrics{"" if test_only else ", ModuleWithFreeze"}, nn.Module):

	def __init__(self, in_channels, out_channels, stride,
		bottleneck_channels, dilation{"" if test_only else ", norm_cls"}):
		CNNMetrics.__init__(self, in_channels, out_channels, stride)
		nn.Module.__init__(self)

		if in_channels == out_channels:
			self.shortcut = lambda x: x
		else:
			self.shortcut = {Conv2d("in_channels", "out_channels", 1, "stride", None, False, None, None, "norm_cls(out_channels)", t=3)}
{"" if test_only else t3+"weight_init.c2_msra_fill(self.shortcut)"+n}\

		self.conv1 = {Conv2d("in_channels", "bottleneck_channels", 1, "stride" if cfg["stride_in_1x1"] else None, None, False, None, None, "norm_cls(bottleneck_channels)", t=2)}
{"" if test_only else t2+"weight_init.c2_msra_fill(self.conv1)"+n+n}\
		self.conv2 = {Conv2d("bottleneck_channels", "bottleneck_channels", 3, None if cfg["stride_in_1x1"] else "stride", "dilation", False, cfg["num_groups"], "dilation", "norm_cls(bottleneck_channels)", t=2)}
{"" if test_only else t2+"weight_init.c2_msra_fill(self.conv2)"+n+n}\
		self.conv3 = {Conv2d("bottleneck_channels", "out_channels", 1, None, None, False, None, None, "norm_cls(out_channels)", t=2)}
{"" if test_only else t2+"weight_init.c2_msra_fill(self.conv3)"+n}\

	def forward(self, x):
		out = F.relu_(self.conv1(x))
		out = F.relu_(self.conv2(out))
		out = F.relu_(self.conv3(out) + self.shortcut(x))
		return out\n\n\n""")

	def generate_ResNetStage():
		res.append("""\
class ResNetStage(CNNMetrics, nn.Sequential):

	def __init__(self, block_class, in_channels, out_channels,
		first_stride, num_blocks, **kwargs):
		CNNMetrics.__init__(self, in_channels, out_channels, first_stride)

		res = [block_class(in_channels, out_channels, first_stride, **kwargs)]
		for i in range(num_blocks-1):
			res.append(block_class(out_channels, out_channels, 1, **kwargs))

		nn.Sequential.__init__(self, *res)
""")

		if not test_only:
			res.append("""
	def freeze(self):
		for block in self.children():
			block.freeze()
""")
		res.append("\n\n")


	def generate_NoBatchNorm():
		res.append("NoBatchNorm = lambda x: None\n\n\n")


	def generate_ResNet():
		res.append(f"""\
class ResNet(nn.Module):

	\"\"\"
	Properties of class:
		stride (dict[str, int]): Произведение всех stride до данного stage
		in_channels (int): Количество каналов на входе
		out_channels (dict[str, int]): Количество каналов на данном stage
		stages_list (list[str]): Непустой список всех stage'ей. Отсортированный
			по порядку выполнения
		out_features (list[str]): Непустой список stage'ей, которые должны
			быть возвращены после вызова forward. Отсортированный по
			порядку выполнения
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

		self.in_channels = in_channels
		self.out_features = [i for i in stages_list if i in out_features] # sort
		self.stages_list = stages_list[:stages_list.index(self.out_features[-1])+1]

		self._register_stages(self._build_net(in_channels, self.stages_list))

	def _build_net(self, in_channels, stages_list):
		res = []

		stage = BasicStem(in_channels, {cfg["stem_out_channels"]}{"" if test_only else ", "+norm})
		res.append(("stem", "stem" in self.out_features, stage))
		if len(stages_list) == 1: return res

""")
		num_blocks_per_stage = {
			18: [2, 2, 2, 2],
			34: [3, 4, 6, 3],
			50: [3, 4, 6, 3],
			101: [3, 4, 23, 3],
			152: [3, 8, 36, 3],
		}[cfg["depth"]]

		block_class = "BasicBlock" if cfg["depth"] in [18, 34] else "BottleneckBlock"

		res.append(f"""\t\tnum_blocks_per_stage = {num_blocks_per_stage}\n""")
		res.append(f"""\t\tout_channels         = {cfg["res2_out_channels"]}\n""")
		if block_class == "BottleneckBlock":
			res.append(f"""\t\tbottleneck_channels  = {cfg["num_groups"] * cfg["width_per_group"]}\n""")

		res.append(f"""
		for i, stage_name in enumerate(stages_list[1:]):
			p = 2**i
			stage = ResNetStage({block_class},
				in_channels=stage.out_channels,
				out_channels=out_channels*p,
				first_stride=1 if i == 0{f" or stage_name == {q}res5{q}" if cfg["res5_dilation"] == 2 else ""} else 2""")
		if block_class == "BottleneckBlock":
			res.append(f""",
				num_blocks=num_blocks_per_stage[i],
				bottleneck_channels=bottleneck_channels*p,
				dilation={f"2 if stage_name == {q}res5{q} else 1" if cfg["res5_dilation"] == 2 else "1"}""")
		if not test_only:
			res.append(f""",
				norm_cls={norm}""")

		res.append(f""")
			res.append((stage_name, stage_name in self.out_features, stage))

		return res

	def _register_stages(self, stages):
		self.stages = stages
		self.stride = {{}}
		self.out_channels = {{}}
		cur_stride = 1

		for name, _, stage in stages:
			self.add_module(name, stage) # register stages

			cur_stride *= stage.stride # Расчет stride и out_channels
			self.stride[name] = cur_stride
			self.out_channels[name] = stage.out_channels

{t+"@torch.no_grad()"+n if test_only else ""}\
	def forward(self, x):
		assert x.dim() == 4 and x.size(1) == self.stem.in_channels,\\
			f"ResNet takes an input of shape (N, {{self.stem.in_channels}}, H, W). Got {{x.shape}} instead!"

		res = {{}}
		for name, is_result, stage in self.stages:
			x = stage(x)
			if is_result: res[name] = x

		return res
""")

		if not test_only:
			res.append("""
	def freeze(self, freeze_at):
		\"\"\"Заморозить первые 'freeze_at' stages.

		Например, если freeze_at = 1, то заморожен будет только "stem".
		\"\"\"
		assert freeze_at in range(0, len(self.stages)+1)
		for _, _, stage in self.stage[:freeze_at]: stage.freeze()
		return self

	def extract(self):
		\"\"\"Подготовить модель для тестирования.

		Заменить все Conv2d на nn.Conv2d, объединив их с BatchNorm
		\"\"\"

		return Conv2d.extract_all(self)
""")



	generate_imports()
	generate_CNNMetrics()
	if not test_only:
		generate_ModuleWithFreeze()
	generate_BasicStem()
	if cfg["depth"] in [18, 34]:
		generate_BasicBlock()
	else:
		generate_BottleneckBlock()
	generate_ResNetStage()
	if not test_only and cfg["norm"] == "None":
		generate_NoBatchNorm()
	generate_ResNet()

	return "".join(res)
