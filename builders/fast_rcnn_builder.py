import os

FOLDER = "parts"


def build_fast_rcnn(cfg, test_only=False, engine="pytorch"):
	os.makedirs(FOLDER, exist_ok=True)

	code, libs = generate_fast_rcnn(cfg, test_only, engine)

	with open(os.path.join(FOLDER, "fast_rcnn.py"), "w") as f:
		f.write(code)

	return libs


def generate_fast_rcnn(cfg, test_only=False, engine="pytorch"):
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

	def assert_FastRCNNConvFCHead(cfg):
		assert cfg["norm"] in ["None", "BN", "FrozenBN"]
		assert isinstance(cfg["conv"], list)
		assert all([isinstance(i, int) and i > 0 for i in cfg["conv"]])
		assert isinstance(cfg["fc"], list)
		assert all([isinstance(i, int) and i > 0 for i in cfg["fc"]])
		assert len(cfg["conv"]) + len(cfg["fc"]) > 0

	assert_FastRCNNConvFCHead(cfg["BOX_HEAD"]["FastRCNNConvFCHead"])

	if engine == "pytorch":
		return generate_fast_rcnn_pytorch(cfg, test_only)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_fast_rcnn_pytorch(cfg, test_only):
	res = []
	libs = set()
	res.append("###  Automatically-generated file  ###\n\n")

	def generate_imports():
		res.append("import torch\n")
		res.append("import torch.nn as nn\n")
		res.append("import torch.nn.functional as F\n\n")

		res.append("from ..layers import ROIPooler")
		libs.add("layers/pooler.py")

		if not test_only:
			need_frozenBN = cfg["BOX_HEAD"]["FastRCNNConvFCHead"] == "FrozenBN"
			res.append(f"""from ..layers import Conv2d{", FrozenBatchNorm2d" if need_frozenBN else ""}\n\n""")
			libs.add("layers/conv_wrapper.py")
			if need_frozenBN:
				libs.add("layers/freeze_batchnorm.py")

		res.append("\n")


	def generate_FastRCNNConvFCHead(lcfg):
		norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[lcfg["norm"]]

		res.append(f"""\
class FastRCNNConvFCHead(nn.Module):
	\"\"\"
	A head with several 3x3 conv layers (each followed by norm & relu) and then
	several fc layers (each followed by relu).
	\"\"\"

	def __init__(self, in_channels, input_size, num_classes, box_dim):
		super().__init__()
		assert isinstance(in_channels, int)
		assert isinstance(input_size, (tuple, list)) and len(input_size) == 2\n\n""")

		in_channels = "in_channels"

		for i, item in enumerate(lcfg["conv"]):
			name = f"self.conv{i+1}"
			if test_only:
				res.append(f"""\
		{name} = nn.Conv2d({in_channels}, {item}, kernel_size=3, padding=1)\n""")
			else:
				res.append(f"""\
		{name} = Conv2d({in_channels}, {item}, kernel_size=3, padding=1{", bias=False" if not test_only and norm != "None" else ""}{", norm="+norm if norm != "None" else ""}, init="c2_msra_fill")\n""")
			in_channels = item

			if i+1 == len(lcfg["conv"]): res.append("\n")

		in_channels = f"{in_channels} * input_size[0] * input_size[1]"

		for i, item in enumerate(lcfg["fc"]):
			res.append(f"""\
		self.fc{i+1} = nn.Linear({in_channels}, {item})\n""")
			in_channels = item

		res.append(f"""
		self.cls_score = nn.Linear({in_channels}, num_classes + 1)
		self.bbox_pred = nn.Linear({in_channels}, {"" if cfg["is_agnostic"] else "num_classes * "}box_dim){"  # agnostic" if cfg["is_agnostic"] else ""}\n""")

		if not test_only:
			res.append("\n")
			for i in range(len(lcfg["fc"])):
				res.append(f"""\
		weight_init.c2_xavier_fill(self.fc{i+1})\n""")

		res.append("""
		nn.init.normal_(self.cls_score.weight, std=0.01)
		nn.init.normal_(self.bbox_pred.weight, std=0.001)
		nn.init.constant_(self.cls_score.bias, 0)
		nn.init.constant_(self.bbox_pred.bias, 0)\n""")

		res.append("""
	def forward(self, x):""")
		for i in range(len(lcfg["conv"])):
			res.append(f"""
		x = F.relu(self.conv{i+1}(x))""")
		res.append("""
		x = torch.flatten(x, start_dim=1)""")
		for i in range(len(lcfg["fc"])):
			res.append(f"""
		x = F.relu(self.fc{i+1}(x))""")
		
		res.append("""\n
		return self.cls_score(x), self.bbox_pred(x)\n""")


	generate_imports()
	if cfg["BOX_HEAD"]["name"] == "FastRCNNConvFCHead":
		generate_FastRCNNConvFCHead(cfg["BOX_HEAD"]["FastRCNNConvFCHead"])

	return "".join(res), libs
