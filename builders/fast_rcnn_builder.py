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
		assert len(cfg["fc"]) > 0

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

		if not test_only:
			need_frozenBN = cfg["BOX_HEAD"]["FastRCNNConvFCHead"] == "FrozenBN"
			res.append(f"""from ..layers import Conv2d{", FrozenBatchNorm2d" if need_frozenBN else ""}\n\n""")
			libs.add("layers/conv_wrapper.py")
			if need_frozenBN:
				libs.add("layers/freeze_batchnorm.py")

		res.append("\n")


	def generate_FastRCNNConvFCHead(cfg):
		norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[cfg["norm"]]

		res.append(f"""\
class FastRCNNConvFCHead(nn.Module):
	\"\"\"
	A head with several 3x3 conv layers (each followed by norm & relu) and then
	several fc layers (each followed by relu).
	\"\"\"

	def __init__(self, input_channel, input_size):
		super().__init__()
		assert isinstance(input_channel, int)
		assert isinstance(input_size, (tuple, list)) and len(input_size) == 2

		self.num_outputs = {cfg["fc"][-1]}\n\n""")

		input_channel = "input_channel"

		for i, item in enumerate(cfg["conv"]):
			name = f"self.conv{i+1}"
			if test_only:
				res.append(f"""\
		{name} = nn.Conv2d({input_channel}, {item}, kernel_size=3, padding=1)\n""")
			else:
				res.append(f"""\
		{name} = Conv2d({input_channel}, {item}, kernel_size=3, padding=1{", bias=False" if not test_only and norm != "None" else ""}{", norm="+norm if norm != "None" else ""}, init="c2_msra_fill")\n""")
			input_channel = item

			if i+1 == len(cfg["conv"]): res.append("\n")

		input_channel = f"{input_channel} * input_size[0] * input_size[1]"

		for i, item in enumerate(cfg["fc"]):
			if i+1 == len(cfg["fc"]): item = "self.num_outputs"
			res.append(f"""\
		self.fc{i+1} = nn.Linear({input_channel}, {item})\n""")
			input_channel = item

		if not test_only:
			res.append("\n")
			for i in range(len(cfg["fc"])):
				res.append(f"""\
		weight_init.c2_xavier_fill(self.fc{i+1})\n""")

		res.append("""
	def forward(self, x):""")
		for i in range(len(cfg["conv"])):
			res.append(f"""
		x = F.relu(self.conv{i+1}(x))""")
		res.append("""
		x = torch.flatten(x, start_dim=1)""")
		for i in range(len(cfg["fc"])):
			res.append(f"""
		x = x = F.relu(self.fc{i+1}(x))""")
		
		res.append("""
		return x\n""")


	generate_imports()
	if cfg["BOX_HEAD"]["name"] == "FastRCNNConvFCHead":
		generate_FastRCNNConvFCHead(cfg["BOX_HEAD"]["FastRCNNConvFCHead"])

	return "".join(res), libs
