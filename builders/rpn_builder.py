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

	is_pos_num = lambda x: isinstance(x, int) and x > 0
	is_procent = lambda x: x >= 0 and x <= 1

	assert cfg["head_name"] in ["StandardRPNHead"]
	assert isinstance(cfg["iou_thresholds"], list) and len(cfg["iou_thresholds"]) == 2
	assert is_procent(cfg["iou_thresholds"][0]) and is_procent(cfg["iou_thresholds"][1])

	ratios = cfg["ANCHOR_GENERATOR"]["ratios"]
	sizes = cfg["ANCHOR_GENERATOR"]["sizes"]
	assert isinstance(ratios, list) and len(ratios) > 0
	assert isinstance(sizes, list) and len(sizes) > 0
	if isinstance(ratios[0], list):
		assert len(ratios[0]) > 0
		assert len(set([len(i) for i in ratios])) == 1
	if isinstance(sizes[0], list):
		assert len(sizes[0]) > 0
		assert len(set([len(i) for i in sizes])) == 1

	assert cfg["bbox_reg_loss_type"] in ["smooth_l1", "giou"]
	assert cfg["LOSS_WEIGHT"]["rpn_cls"] >= 0
	assert cfg["LOSS_WEIGHT"]["rpn_loc"] >= 0

	assert is_pos_num(cfg["TRAIN"]["pre_topk"])
	assert is_procent(cfg["TRAIN"]["nms_thress"])
	assert is_pos_num(cfg["TRAIN"]["post_topk"])
	assert is_pos_num(cfg["TRAIN"]["batch_size_per_image"])
	assert is_procent(cfg["TRAIN"]["positive_fraction"])

	assert is_pos_num(cfg["TEST"]["pre_topk"])
	assert is_procent(cfg["TEST"]["nms_thress"])
	assert is_pos_num(cfg["TEST"]["post_topk"])

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
		res.append("import torch.nn.functional as F\n")
		res.append("import torchvision.ops.boxes as box_ops\n\n")

		res.append(f"from ..utils import Boxes, MultiAnchors{'' if test_only else ', Matcher, Subsampler'}\n\n")
		libs.add("utils/boxes.py")
		libs.add("utils/anchors.py")
		if not test_only:
			libs.add("utils/matcher.py")
			libs.add("utils/subsampler.py")

	def generate_StandardRPNHead():
		res.append("""
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
		return [self.logits_reshape(self.objectness_logits(x)) for x in features],\\
			[self.deltas_reshape(self.anchor_deltas(x)) for x in features]\n\n""")

	def generate_SelectRPNProposals():
		res.append("""
class SelectRPNProposals(nn.Module):

	def __init__(self, pre_topk, nms_thresh, post_topk, min_box_size):
		super().__init__()
		self.pre_topk = pre_topk
		self.nms_thresh = nms_thresh
		self.post_topk = post_topk
		self.min_box_size = min_box_size\n""")

		res.append("""
	def forward(self, proposals, logits, image_sizes):
		device = proposals[0].device
		batch_idx = torch.arange(len(proposals[0]), device=device).unsqueeze(-1)

		# 1. Select top-k anchor for every level and every image
		topk_scores = []
		topk_proposals = []
		level_ids = []

		for i in range(len(proposals)):
			num_proposals = min(self.pre_topk, logits[i].shape[1])
			logits_i, idx = logits[i].sort(descending=True, dim=1)

			topk_scores.append(logits_i[:, :num_proposals])  # (B, topk)
			topk_proposals.append(proposals[i][batch_idx, idx[:, :num_proposals]])  # (B, topk, 4)
			level_ids.append(torch.full((num_proposals,), i, dtype=torch.int64, device=device))

		# 2. Concat all levels together
		topk_scores = torch.cat(topk_scores, dim=1)  # (B, A)
		topk_proposals = torch.cat(topk_proposals, dim=1)  # (B, A, 4)
		level_ids = torch.cat(level_ids, dim=0)  # (A,)

		# 3. For each image, run a per-level NMS, and choose topk results.
		res = []
		for i in range(len(image_sizes)):
			scores_i, proposals_i, levels_i = topk_scores[i], topk_proposals[i], level_ids

			keep = torch.isfinite(proposals_i).all(dim=1) & torch.isfinite(scores_i)
			if not keep.all():""")
		if not test_only:
			res.append("""
				if self.training:
					raise FloatingPointError("Predicted boxes or scores contain Inf/NaN. Training has diverged.")""")
		res.append("""
				scores_i, proposals_i, levels_i = scores_i[keep], proposals_i[keep], levels_i[keep]
			
			Boxes.clamp_(proposals_i, image_sizes[i])

			# filter empty boxes
			keep = Boxes.nonempty(proposals_i, threshold=self.min_box_size)
			if not keep.all():
				proposals_i, scores_i, levels_i = proposals_i[keep], scores_i[keep], levels_i[keep]

			keep = box_ops.batched_nms(proposals_i, scores_i, levels_i, self.nms_thresh)  # TODO: Is it really faster?
			keep = keep[:self.post_topk]  # keep is already sorted
			res.append((scores_i[keep], proposals_i[keep]))

		return res\n\n""")


	def generate_RPN():
		res.append(f"""
class RPN(nn.Module):

	def __init__(self, in_channels, strides):
		super().__init__()

		self.anchor_generator = MultiAnchors({cfg["ANCHOR_GENERATOR"]["sizes"]}, {cfg["ANCHOR_GENERATOR"]["ratios"]}, strides)
		assert len(set(self.anchor_generator.num_anchors)) == 1
		self.rpn_head = StandardRPNHead(in_channels, self.anchor_generator.num_anchors[0], box_dim=4)""")
		if test_only:
			res.append(f"""
		self.find_top_rpn_proposals = SelectRPNProposals({cfg["TEST"]["pre_topk"]}, {cfg["TEST"]["nms_thress"]}, {cfg["TEST"]["post_topk"]}, min_box_size={cfg["min_box_size"]})\n""")
		else:
			res.append(f"""
		self.anchor_matcher = Matcher(bg_threshold={cfg["iou_thresholds"][0]}, fg_threshold={cfg["iou_thresholds"][1]}, allow_low_quality_matches=True)
		self.subsampler = Subsampler(num_samples={cfg["TRAIN"]["batch_size_per_image"]}, positive_fraction={cfg["TRAIN"]["positive_fraction"]})

		selector_test = SelectRPNProposals({cfg["TEST"]["pre_topk"]}, {cfg["TEST"]["nms_thress"]}, {cfg["TEST"]["post_topk"]}, min_box_size={cfg["min_box_size"]})
		selector_train = SelectRPNProposals({cfg["TRAIN"]["pre_topk"]}, {cfg["TRAIN"]["nms_thress"]}, {cfg["TRAIN"]["post_topk"]}, min_box_size={cfg["min_box_size"]})
		self.find_top_rpn_proposals = lambda *args: (selector_train if self.training else selector_test)(*args)\n""")

		if not test_only:
			res.append(f"""
		self.loss_weight = {cfg["LOSS_WEIGHT"]}\n""")

		if not test_only:
			res.append(f"""
	@torch.no_grad()
	def label_and_sample_anchors(self, anchors, gt_instances):
		anchors = torch.cat([a.get_xyxy() for a in anchors])

		gt_labels = []
		gt_boxes = []  # matched ground truth boxes
		for item in [x["boxes"] for x in gt_instances]:
			matched_idxs, gt_labels_i = self.anchor_matcher(item, anchors)
			gt_labels.append(self.subsampler.return_as_mask(gt_labels_i))  # (N, A*H*W)
			gt_boxes.append(item[matched_idxs] if len(item) != 0 else torch.zeros_like(anchors))

		return gt_labels, gt_boxes

	def losses(self, anchors, gt_instances, pred_objectness_logits, pred_anchor_deltas):
		gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
		losses = self.losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes)
		return {{k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}}\n""")

		res.append(f"""
	def forward(self, features, image_sizes{"" if test_only else ", gt_instances=None"}):
		\"\"\"
		Args:
			features (list[Tensor]): input features
			image_sizes (list[int, int]): sizes of original inputs""")

		if not test_only:
			res.append("""
			gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
				Each `Instances` stores ground-truth instances for the corresponding image.""")

		res.append("""\n
		Returns:
			proposals: list[Tensor]: list of proposal boxes of shape [Ai, 4]""")

		if not test_only:
			res.append("""
			loss: dict[Tensor] or None""")
		res.append("""
		\"\"\"""")

		res.append("""
		pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
		anchors = self.anchor_generator([f.shape[-2:] for f in features])
		del features\n""")

		if not test_only:
			res.append("""
		if gt_instances is not None:
			losses = self.losses(anchors, gt_instances, pred_objectness_logits, pred_anchor_deltas)
		else:
			losses = {}\n""")

		res.append(f"""
		# decode proposals and choose the best ones
		pred_proposals = [a.apply_deltas(d) for a, d in zip(anchors, pred_anchor_deltas)]
		proposals = self.find_top_rpn_proposals(pred_proposals, pred_objectness_logits, image_sizes)
		return proposals{"" if test_only else ", losses"}""")

	generate_imports()
	generate_StandardRPNHead()
	generate_SelectRPNProposals()
	generate_RPN()

	return "".join(res), libs
