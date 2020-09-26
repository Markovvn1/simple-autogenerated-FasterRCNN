import os

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
		cfg (Dict[String, Any]): Список параметров конфигурации RPN:
			iou_thresholds: [0.3, 0.7]  # Пороги для определения является ли anchor background или foreground
			min_box_size: 0
			ANCHOR_GENERATOR:  # Параметры генератора anchors
				ratios: [0.5, 1.0, 2.0]
				sizes: [[32], [64], [128], [256], [512]]
			LOSS:
				global_weight: 2.0  # Вклад ошибки RPN модуля в общую ошибку сети
				box_reg_weight: 0.5  # Вклад ошибки локализации в ошибку RPN модуля
				bbox_reg_loss_type: "giou"  # Тип ошибки для уточнения смещения. {"smooth_l1", "giou"}
				smooth_l1_beta: 1.0  # Используется только если bbox_reg_loss_type == "smooth_l1"
			TRAIN:  # Парамеры для обучения
				pre_topk: 2000
				nms_thress: 0.7
				post_topk: 1000
				batch_size_per_image: 256  # Количество изображений используемых для обучения RPN
				positive_fraction: 0.5
			TEST:  # Параметры для тестирования
				pre_topk: 1000
				nms_thress: 0.7
				post_topk: 1000

		test_only (Bool): Нужно ли генерировать код только для тестирования, или он
			будет использоваться и для обучения

		engine (String): Движок для которого будет сгенерирован код. Возможные значения:
			"pytorch"
	"""

	assert isinstance(test_only, bool)
	assert isinstance(engine, str)

	is_pos_num = lambda x: isinstance(x, int) and x > 0
	is_procent = lambda x: x >= 0 and x <= 1

	assert isinstance(cfg["TRAIN"]["iou_thresholds"], list) and len(cfg["TRAIN"]["iou_thresholds"]) == 2
	assert is_procent(cfg["TRAIN"]["iou_thresholds"][0]) and is_procent(cfg["TRAIN"]["iou_thresholds"][1])

	ratios = cfg["ANCHOR_GENERATOR"]["ratios"]
	sizes = cfg["ANCHOR_GENERATOR"]["sizes"]
	assert isinstance(ratios, list) and len(ratios) > 0
	assert isinstance(sizes, list) and len(sizes) > 0
	if isinstance(ratios[0], list):
		assert len(ratios[0]) > 0
		assert len(set(len(i) for i in ratios)) == 1
	if isinstance(sizes[0], list):
		assert len(sizes[0]) > 0
		assert len(set(len(i) for i in sizes)) == 1

	assert cfg["LOSS"]["bbox_reg_loss_type"] in ["smooth_l1", "giou"]
	assert cfg["LOSS"]["global_weight"] >= 0
	assert is_procent(cfg["LOSS"]["box_reg_weight"])
	assert cfg["LOSS"]["smooth_l1_beta"] >= 0

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
		res.append("import torchvision.ops.boxes as box_ops\n")
		res.append(f"from fvcore.nn import {'smooth_l1_loss' if cfg['LOSS']['bbox_reg_loss_type'] == 'smooth_l1' else 'giou_loss'}\n\n")

		res.append(f"from ..utils import Boxes, Anchors, BoxTransform, MultiAnchors{'' if test_only else ', Matcher, Subsampler'}\n\n")
		libs.add("utils/boxes.py")
		libs.add("utils/anchors.py")
		libs.add("utils/box_transform.py")
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
class SelectRPNProposals:

	def __init__(self, pre_topk, nms_thresh, post_topk, min_box_size):
		super().__init__()
		self.pre_topk = pre_topk
		self.nms_thresh = nms_thresh
		self.post_topk = post_topk
		self.min_box_size = min_box_size\n""")

		res.append("""
	def __call__(self, logits, proposals, image_sizes):
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
				raise FloatingPointError("Predicted boxes or scores contain Inf/NaN. Training has diverged.")\n""")
		else:
			res.append("""
				scores_i, proposals_i, levels_i = scores_i[keep], proposals_i[keep], levels_i[keep]\n""")

		res.append("""
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
		self.transform = BoxTransform(weights={cfg["box_transform_weights"]})
		assert len(set(self.anchor_generator.num_anchors)) == 1
		self.rpn_head = StandardRPNHead(in_channels, self.anchor_generator.num_anchors[0], box_dim=4)""")
		if test_only:
			res.append(f"""
		self.find_top_proposals = SelectRPNProposals({cfg["TEST"]["pre_topk"]}, {cfg["TEST"]["nms_thress"]}, {cfg["TEST"]["post_topk"]}, min_box_size={cfg["min_box_size"]})\n""")
		else:
			res.append(f"""
		self.anchor_matcher = Matcher(bg_threshold={cfg["TRAIN"]["iou_thresholds"][0]}, fg_threshold={cfg["TRAIN"]["iou_thresholds"][1]}, allow_low_quality_matches=True)
		self.subsampler = Subsampler(num_samples={cfg["TRAIN"]["batch_size_per_image"]}, positive_fraction={cfg["TRAIN"]["positive_fraction"]}, bg_label=0)

		selector_test = SelectRPNProposals({cfg["TEST"]["pre_topk"]}, {cfg["TEST"]["nms_thress"]}, {cfg["TEST"]["post_topk"]}, min_box_size={cfg["min_box_size"]})
		selector_train = SelectRPNProposals({cfg["TRAIN"]["pre_topk"]}, {cfg["TRAIN"]["nms_thress"]}, {cfg["TRAIN"]["post_topk"]}, min_box_size={cfg["min_box_size"]})
		self.find_top_proposals = lambda *args: (selector_train if self.training else selector_test)(*args)\n""")

		if not test_only:
			res.append(f"""
		self.loss_weight = {{"rpn_cls": {round(cfg["LOSS"]["global_weight"] * (1-cfg["LOSS"]["box_reg_weight"]), 7)}, "rpn_loc": {round(cfg["LOSS"]["global_weight"] * cfg["LOSS"]["box_reg_weight"], 7)}}}\n""")

		if not test_only:
			pred_data = "pred_anchor_deltas" if cfg["LOSS"]["bbox_reg_loss_type"] == "smooth_l1" else "pred_proposals"

			res.append(f"""
	@torch.no_grad()
	def label_and_sample_anchors(self, anchors, targets):
		\"\"\"
		Каждому anchor подобрать наиболее подходящий ground truth box.

		Returns:
			(tensor): Массив, показывающий является ли данных anchor foreground(1)
				или background(0). (A,)
			(tensor): Массив с ground truth коробками для каждого anchor. (A, 4)
		\"\"\"

		gt_labels = []
		gt_boxes = []  # matched ground truth boxes

		for item in [x["boxes"] for x in targets]:
			matched_idxs, gt_labels_i = self.anchor_matcher(item, Boxes.xywh2xyxy(anchors))
			gt_labels.append(self.subsampler.return_as_mask(gt_labels_i))  # (N, A*H*W)
			gt_boxes.append(item[matched_idxs] if len(item) != 0 else torch.zeros_like(anchors))

		return torch.stack(gt_labels), torch.stack(gt_boxes)

	def losses(self, anchors, targets, pred_objectness_logits, {pred_data}):
		assert targets is not None

		anchors = torch.cat(anchors)
		gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, targets)

		pos_mask = gt_labels == 1
		num_pos_anchors = pos_mask.sum().item()
		num_neg_anchors = (gt_labels == 0).sum().item()

		{pred_data} = torch.cat({pred_data}, dim=1)
		pred_objectness_logits = torch.cat(pred_objectness_logits, dim=1)

		losses = {{}}\n""")

			if cfg["LOSS"]["bbox_reg_loss_type"] == "smooth_l1":
				res.append(f"""
		gt_anchor_deltas = self.transform.get_deltas(anchors, gt_boxes) # (N, sum(Hi*Wi*Ai), 4 or 5)
		losses["rpn_loc"] = smooth_l1_loss(
			{pred_data}[pos_mask], gt_anchor_deltas[pos_mask],
			beta={cfg["LOSS"]["smooth_l1_beta"]}, reduction="sum",
		)\n""")
			elif cfg["LOSS"]["bbox_reg_loss_type"] == "giou":
				res.append(f"""
		losses["rpn_loc"] = giou_loss(
			{pred_data}[pos_mask], gt_boxes[pos_mask],
			reduction="sum",
		)\n""")

			res.append("""
		valid_mask = gt_labels >= 0
		losses["rpn_cls"] = F.binary_cross_entropy_with_logits(
			pred_objectness_logits[valid_mask],
			gt_labels[valid_mask].float(),
			reduction="sum",
		)

		normalizer = self.subsampler.num_samples * len(gt_labels)
		return {k: v * self.loss_weight[k] / normalizer for k, v in losses.items()}\n""")

		res.append(f"""
	def forward(self, features, image_sizes{"" if test_only else ", targets=None"}):
		\"\"\"
		Args:
			features (list[Tensor]): input features
			image_sizes (list[int, int]): sizes of original inputs""")

		if not test_only:
			res.append("""
			targets (list[Instances], optional): a length `N` list of `Instances`s.
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
		del features

		# decode proposals
		proposals = [self.transform.apply_deltas(a, d) for a, d in zip(anchors, pred_anchor_deltas)]
		""")

		if not test_only:
			res.append(f"""
		if targets is not None:
			losses = self.losses(anchors, targets, pred_objectness_logits, {"pred_anchor_deltas" if cfg["LOSS"]["bbox_reg_loss_type"] == "smooth_l1" else "proposals"})
		else:
			losses = {{}}\n""")

		res.append(f"""
		# choose the best proposals
		proposals = self.find_top_proposals(pred_objectness_logits, proposals, image_sizes)
		return proposals{"" if test_only else ", losses"}""")

	generate_imports()
	generate_StandardRPNHead()
	generate_SelectRPNProposals()
	generate_RPN()

	return "".join(res), libs
