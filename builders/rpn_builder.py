import os

# Важно!
# 1. Перед вызовом SelectRPNProposals: boxes.clip(image_size)
# 2. Проверки (assert), которые были в оригинальном StandardRPNHead

FOLDER = "parts"


def build_rpn(cfg, test_only=False, engine="pytorch"):
	os.makedirs(FOLDER, exist_ok=True)

	code, libs = generate_rpn(cfg, test_only, engine)

	with open(os.path.join(FOLDER, "rpn.py"), "w") as f:
		f.write(code)

	return libs


def generate_rpn(cfg, test_only=False, engine="pytorch"):
	"""
	Генерирует код для RPN используя параметры из cfg.

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
	assert isinstance(engine, str)

	assert cfg["head_name"] in ["StandardRPNHead"]

	if engine == "pytorch":
		return generate_rpn_pytorch(cfg, test_only)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_rpn_pytorch(cfg, test_only):
	res = []
	libs = set()
	res.append("###  Automatically-generated file  ###\n\n")

	def generate_imports():
		res.append("import torch\n")
		res.append("import torch.nn as nn\n")
		res.append("import torch.nn.functional as F\n\n")

	def generate_StandardRPNHead():
		res.append("""\
class StandardRPNHead(nn.Module):

	\"\"\"
	Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
	Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
	objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
	specifying how to deform each anchor into an object proposal.
	\"\"\"

	def __init__(self, in_channels, num_anchors, box_dim):
		super().__init__()

		self.in_channels = in_channels
		self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
		self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
		self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1)

		# (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
		self.logits_reshape = lambda x: x.permute(0, 2, 3, 1).flatten(1)
		# (N, A*D, Hi, Wi) -> (N, A, D, Hi, Wi) -> (N, Hi, Wi, A, D) -> (N, Hi*Wi*A, D)
		self.deltas_reshape = lambda x: x.view(-1, num_anchors, box_dim, x.size(-2), x.size(-1)).permute(0, 3, 4, 1, 2).flatten(1, -2)\n""")

		if not test_only:
			res.append("""
		for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
			nn.init.normal_(l.weight, std=0.01)
			nn.init.constant_(l.bias, 0)\n""")

		res.append("""
	def forward(self, features):
		for x in features: assert (x.dim() == 4) and (x.size(1) == self.in_channels)

		features = [F.relu(self.conv(x)) for x in features]
		return [self.logits_reshape(self.objectness_logits(x)) for x in features],
			[self.deltas_reshape(self.anchor_deltas(x)) for x in features]\n\n""")

	generate_imports()
	generate_StandardRPNHead()

	return "".join(res), libs
