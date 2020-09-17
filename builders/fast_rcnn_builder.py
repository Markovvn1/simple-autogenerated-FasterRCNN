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

	is_pos_num = lambda x: isinstance(x, int) and x > 0
	is_procent = lambda x: isinstance(x, float) and x >= 0 and x <= 1

	assert_FastRCNNConvFCHead(cfg["BOX_HEAD"]["FastRCNNConvFCHead"])

	assert isinstance(cfg["is_agnostic"], bool)
	
	assert isinstance(cfg["box_transform_weights"], (list, tuple)) and len(cfg["box_transform_weights"]) == 4
	assert all([isinstance(i, float) for i in cfg["box_transform_weights"]])
	
	assert isinstance(cfg["TRAIN"]["iou_thresholds"], (list, tuple)) and len(cfg["TRAIN"]["iou_thresholds"]) == 2
	assert all([is_procent(i) for i in cfg["TRAIN"]["iou_thresholds"]])

	assert isinstance(cfg["LOSS"]["global_weight"], float)
	assert is_procent(cfg["LOSS"]["box_reg_weight"])
	assert cfg["LOSS"]["bbox_reg_loss_type"] in ["smooth_l1", "giou"]

	assert is_pos_num(cfg["TRAIN"]["batch_size_per_image"])
	assert is_procent(cfg["TRAIN"]["positive_fraction"])
	assert isinstance(cfg["TRAIN"]["append_gt_to_proposal"], bool)

	assert is_procent(cfg["TEST"]["nms_thresh"])
	assert is_procent(cfg["TEST"]["score_thresh"])

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

		res.append("from ..layers import RoIPooler")
		libs.add("layers/pooler.py")

		if not test_only:
			if len(cfg["BOX_HEAD"]["FastRCNNConvFCHead"]["conv"]) > 0:
				res.append(", Conv2d")
				libs.add("layers/conv_wrapper.py")

			if cfg["BOX_HEAD"]["FastRCNNConvFCHead"]["norm"] == "FrozenBN":
				res.append(", FrozenBatchNorm2d")
				libs.add("layers/freeze_batchnorm.py")

		res.append("\nfrom ..utils import Boxes")
		libs.add("utils/boxes.py")

		if not test_only:
			res.append(", Matcher, BoxTransform")
			libs.add("utils/matcher.py")
			libs.add("utils/box_transform.py")

		res.append("\n\n")


	def generate_FastRCNNConvFCHead(lcfg):
		norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[lcfg["norm"]]

		res.append(f"""
class FastRCNNConvFCHead(nn.Module):
	\"\"\"
	A head with several 3x3 conv layers (each followed by norm & relu) and then
	several fc layers (each followed by relu).
	\"\"\"

	def __init__(self, in_channels, input_size, num_classes, box_dim):
		super().__init__()
		assert isinstance(in_channels, int) and in_channels > 0
		assert isinstance(num_classes, int) and num_classes > 0
		if isinstance(input_size, int): input_size = (input_size, input_size)
		assert isinstance(input_size, (tuple, list)) and len(input_size) == 2
		assert isinstance(input_size[0], int) and isinstance(input_size[1], int)\n\n""")

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
		self.bbox_pred = nn.Linear({in_channels}, {"" if cfg["is_agnostic"] else "num_classes * "}box_dim){"  # agnostic" if cfg["is_agnostic"] else ""}
		self.bbox_pred_reshape = lambda x: x.view(-1, {"1" if cfg["is_agnostic"] else "num_classes"}, box_dim)\n""")

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
		return self.cls_score(x), self.bbox_pred_reshape(self.bbox_pred(x))\n\n""")


	def generate_SelectRCNNPredictions():
		res.append(f"""
class SelectRCNNPredictions:

	def __init__(self, score_thresh, nms_thresh, topk_per_image):
		assert score_thresh >= 0 and score_thresh <= 1
		assert nms_thresh >= 0 and nms_thresh <= 1

		self.score_thresh = score_thresh
		self.nms_thresh = nms_thresh
		self.topk_per_image = topk_per_image

	def _process_image(self, scores, boxes, image_size):
		valid_mask = torch.isfinite(boxes).all(dim=(1, 2)) & torch.isfinite(scores).all(dim=1)
		if not valid_mask.all():
			boxes = boxes[valid_mask]
			scores = scores[valid_mask]

		scores = scores[:, :-1]  # cut class for background
		Boxes.clip_(boxes, image_size)

		# filter_inds contain 2 elements: (idx of predictions, class)
		filter_inds = (scores > self.score_thresh).nonzero().unbind(1)
		boxes = boxes[filter_inds{"[0], 0" if cfg["is_agnostic"] else ""}]{"  # agnostic" if cfg["is_agnostic"] else ""}
		scores = scores[filter_inds]

		# Apply per-class NMS
		keep = batched_nms(boxes, scores, filter_inds[1], self.nms_thresh)
		if self.topk_per_image >= 0:
			keep = keep[:self.topk_per_image]

		return {{"pred_boxes": boxes[keep], "scores": scores[keep], "pred_classes": filter_inds[1][keep]}}

	def __call__(self, scores, boxes, image_sizes):
		return [self._process_image(*items) for items in zip(scores, boxes, image_sizes)]\n\n""")

	def generate_FastRCNNHead():
		res.append(f"""
class FastRCNNHead(nn.Module):

	def __init__(self, in_channels, strides, num_classes, score_thresh={cfg["TEST"]["score_thresh"]}, nms_thresh={cfg["TEST"]["nms_thresh"]}, topk_per_image=100):
		super().__init__()

		self.box_pooler = RoIPooler({cfg["POOLER"]["resolution"]}, strides, sampling_ratio={cfg["POOLER"]["sampling_ratio"]})
		self.box_head = FastRCNNConvFCHead(in_channels, {cfg["POOLER"]["resolution"]}, num_classes, box_dim=4)
		self.transform = BoxTransform(weights={cfg["box_transform_weights"]})\n""")
		if not test_only:
			res.append(f"""\
		self.matcher = Matcher(bg_threshold={cfg["TRAIN"]["iou_thresholds"][0]}, fg_threshold={cfg["TRAIN"]["iou_thresholds"][1]}, allow_low_quality_matches=False)\n""")

		res.append("""\
		self.find_top_predictions = SelectRoIPredictions(score_thresh, nms_thresh, topk_per_image)

	def _concat_proposals(self, proposals):
		\"\"\"Prepare proposals for pooling. Concatenate it with batch index\"\"\"
		batch_idx = [torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(proposals)]
		proposals = [torch.cat(item, dim=1) for item in zip(batch_idx, proposals)]
		return torch.cat(proposals)  # (idx, x0, y0, x1, y1)

	def forward(self, features, image_sizes, proposals):
		num_prop_per_image = [len(p) for p in proposals]
		
		proposals = self._concat_proposals(proposals)  # (M, 5)
		features = self.box_pooler(features, proposals)

		scores, proposal_deltas = self.box_head(features)
		scores = F.softmax(scores, dim=-1).split(num_prop_per_image)
		boxes = self.transform.apply_deltas(proposals[:, 1:], proposal_deltas).split(num_prop_per_image)

		return self.find_top_predictions(scores, boxes, image_sizes)\n""")

	generate_imports()
	if cfg["BOX_HEAD"]["name"] == "FastRCNNConvFCHead":
		generate_FastRCNNConvFCHead(cfg["BOX_HEAD"]["FastRCNNConvFCHead"])
	generate_SelectRCNNPredictions()
	generate_FastRCNNHead()

	return "".join(res), libs
