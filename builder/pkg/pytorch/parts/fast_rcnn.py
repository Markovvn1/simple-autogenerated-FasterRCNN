from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"fast_rcnn.FastRCNNConvFCHead", "fast_rcnn.SelectRCNNPredictions", "fast_rcnn.FastRCNNHead"})

	def _assert_cfg(self, module, cfg):
		if module == "fast_rcnn.FastRCNNConvFCHead":
			assert cfg["norm"] in ["None", "BN", "FrozenBN"]
			assert isinstance(cfg["conv"], list)
			assert all([isinstance(i, int) and i > 0 for i in cfg["conv"]])
			assert isinstance(cfg["fc"], list)
			assert all([isinstance(i, int) and i > 0 for i in cfg["fc"]])
			assert len(cfg["conv"]) + len(cfg["fc"]) > 0
		elif module == "fast_rcnn.SelectRCNNPredictions":
			assert isinstance(cfg["is_agnostic"], bool)
		elif module == "fast_rcnn.FastRCNNHead":
			is_pos_num = lambda x: isinstance(x, int) and x > 0
			is_procent = lambda x: isinstance(x, float) and x >= 0 and x <= 1

			assert isinstance(cfg["is_agnostic"], bool)
			
			assert isinstance(cfg["box_transform_weights"], (list, tuple)) and len(cfg["box_transform_weights"]) == 4
			assert all([isinstance(i, float) for i in cfg["box_transform_weights"]])
			
			assert isinstance(cfg["TRAIN"]["iou_thresholds"], (list, tuple))
			assert len(cfg["TRAIN"]["iou_thresholds"]) == 2
			assert all([is_procent(i) for i in cfg["TRAIN"]["iou_thresholds"]])
			assert cfg["TRAIN"]["iou_thresholds"][0] <= cfg["TRAIN"]["iou_thresholds"][1]

			assert isinstance(cfg["LOSS"]["global_weight"], float)
			assert is_procent(cfg["LOSS"]["box_reg_weight"])
			assert cfg["LOSS"]["bbox_reg_loss_type"] in ["smooth_l1", "giou"]

			assert is_pos_num(cfg["TRAIN"]["batch_size_per_image"])
			assert is_procent(cfg["TRAIN"]["positive_fraction"])
			assert isinstance(cfg["TRAIN"]["append_gt_to_proposal"], bool)

			assert is_procent(cfg["TEST"]["nms_thresh"])
			assert is_procent(cfg["TEST"]["score_thresh"])

	def _dependencies(self, module, global_params, cfg):
		if module == "fast_rcnn.FastRCNNConvFCHead":
			res = {}
			if global_params["mode"] == "train" and len(cfg["conv"]) > 0:
				res["conv_wrapper.Conv2d"] = None
			if cfg["norm"] == "FrozenBN":
				res["freeze_batchnorm.FrozenBatchNorm2d"] = None
			return res
		if module == "fast_rcnn.SelectRCNNPredictions":
			return {"boxes.clamp_": None}
		if module == "fast_rcnn.FastRCNNHead":
			is_agnostic = {"is_agnostic": cfg["is_agnostic"]}
			res = {"pooler.RoIPooler": cfg["POOLER"], "boxes.xyxy2xywh": None,
				"box_transform.BoxTransform": None, "fast_rcnn.SelectRCNNPredictions": is_agnostic,
				f"fast_rcnn.{cfg['BOX_HEAD']['name']}Head": {**cfg["BOX_HEAD"][cfg["BOX_HEAD"]["name"]], **is_agnostic}}
			if global_params["mode"] == "train":
				res["matcher.Matcher"] = None
				res["subsampler.Subsampler"] = None
			return res

	def _init_file(self, dep):
		if "fast_rcnn.FastRCNNHead" in dep: return "from .fast_rcnn import FastRCNNHead"
		return None

	def _generate(self, global_params, dep, childs):
		res = []
		res.append("""\
import torch
import torch.nn as nn\n""")

		if "fast_rcnn.FastRCNNConvFCHead" in dep or "fast_rcnn.FastRCNNHead" in dep:
			res.append("import torch.nn.functional as F\n")
		if "fast_rcnn.SelectRCNNPredictions" in dep:
			res.append("import torchvision.ops.boxes as box_ops\n")

		fvcore = []
		if global_params["mode"] == "train" and "fast_rcnn.FastRCNNHead" in dep:
			fvcore.append(f"{dep['fast_rcnn.FastRCNNHead']['LOSS']['bbox_reg_loss_type']}_loss")
		if global_params["mode"] == "train" and "fast_rcnn.FastRCNNConvFCHead" in dep:
			fvcore.append("weight_init")

		if fvcore:
			res.append(f"from fvcore.nn import {', '.join(fvcore)}\n")

		first = True

		utils = {"box_transform.BoxTransform", "matcher.Matcher", "subsampler.Subsampler"} & childs.keys()
		if [1 for i in childs.keys() if i[:i.find(".")] == "boxes"]: utils.add("boxes.Boxes")
		if utils:
			first = False
			res.append("\nfrom ..utils import " + ", ".join([i[i.find(".")+1:] for i in utils]) + "\n")

		layers = {"conv_wrapper.Conv2d", "freeze_batchnorm.FrozenBatchNorm2d", "pooler.RoIPooler"} & childs.keys()
		if layers:
			if first: res.append("\n")
			res.append("from ..layers import " + ", ".join([i[i.find(".")+1:] for i in layers]) + "\n")


		if "fast_rcnn.FastRCNNConvFCHead" in dep:
			res.extend(_generate_FastRCNNConvFCHead(global_params, dep["fast_rcnn.FastRCNNConvFCHead"]))
		if "fast_rcnn.SelectRCNNPredictions" in dep:
			res.extend(_generate_SelectRCNNPredictions(global_params, dep["fast_rcnn.SelectRCNNPredictions"]))
		if "fast_rcnn.FastRCNNHead" in dep:
			res.extend(_generate_FastRCNNHead(global_params, dep["fast_rcnn.FastRCNNHead"]))

		return "".join(res)


def _generate_FastRCNNConvFCHead(global_params, cfg):
	mode = global_params["mode"]
	norm = {"None": "None", "BN": "nn.BatchNorm2d", "FrozenBN": "FrozenBatchNorm2d"}[cfg["norm"]]

	res = []
	res.append(f"""\n
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

	for i, item in enumerate(cfg["conv"]):
		name = f"self.conv{i+1}"
		if mode == "test":
			res.append(f"""\
		{name} = nn.Conv2d({in_channels}, {item}, kernel_size=3, padding=1)\n""")
		else:
			res.append(f"""\
		{name} = Conv2d({in_channels}, {item}, kernel_size=3, padding=1\
{", bias=False" if mode == "train" and norm != "None" else ""}\
{", norm="+norm if norm != "None" else ""}, init="c2_msra_fill")\n""")
		in_channels = item

		if i+1 == len(cfg["conv"]): res.append("\n")

	in_channels = f"{in_channels} * input_size[0] * input_size[1]"

	for i, item in enumerate(cfg["fc"]):
		res.append(f"""\
		self.fc{i+1} = nn.Linear({in_channels}, {item})\n""")
		in_channels = item

	res.append(f"""
		self.cls_score = nn.Linear({in_channels}, num_classes + 1)
		self.bbox_pred = nn.Linear({in_channels}, {"" if cfg["is_agnostic"] else "num_classes * "}box_dim){"  # agnostic" if cfg["is_agnostic"] else ""}
		self.bbox_pred_reshape = lambda x: x.view(-1, {"1" if cfg["is_agnostic"] else "num_classes"}, box_dim)\n""")

	if mode == "train":
		res.append("\n")
		for i in range(len(cfg["fc"])):
			res.append(f"""\
		weight_init.c2_xavier_fill(self.fc{i+1})\n""")

		res.append("""
		nn.init.normal_(self.cls_score.weight, std=0.01)
		nn.init.normal_(self.bbox_pred.weight, std=0.001)
		nn.init.constant_(self.cls_score.bias, 0)
		nn.init.constant_(self.bbox_pred.bias, 0)\n""")

	res.append("""
	def forward(self, x):""")
	for i in range(len(cfg["conv"])):
		res.append(f"""
		x = F.relu(self.conv{i+1}(x))""")
	res.append("""
		x = torch.flatten(x, start_dim=1)""")
	for i in range(len(cfg["fc"])):
		res.append(f"""
		x = F.relu(self.fc{i+1}(x))""")
		
	res.append("""\n
		return self.cls_score(x), self.bbox_pred_reshape(self.bbox_pred(x))\n""")
	return res


def _generate_SelectRCNNPredictions(global_params, cfg):
	return [f"""\n
class SelectRCNNPredictions:

	def __init__(self, score_thresh, nms_thresh, topk_per_image):
		assert score_thresh >= 0 and score_thresh <= 1
		assert nms_thresh >= 0 and nms_thresh <= 1

		self.score_thresh = score_thresh
		self.nms_thresh = nms_thresh
		self.topk_per_image = topk_per_image

	def _process_image(self, scores, boxes, image_size):
		valid_mask = torch.isfinite(boxes).all(1).all(1) & torch.isfinite(scores).all(dim=1)
		if not valid_mask.all():
			boxes = boxes[valid_mask]
			scores = scores[valid_mask]

		scores = scores[:, :-1]  # cut class for background
		Boxes.clamp_(boxes, image_size)

		# filter_inds contain 2 elements: (idx of predictions, class)
		filter_inds = (scores > self.score_thresh).nonzero(as_tuple=True)
		boxes = boxes[filter_inds{"[0], 0" if cfg["is_agnostic"] else ""}]{"  # agnostic" if cfg["is_agnostic"] else ""}
		scores = scores[filter_inds]

		# Apply per-class NMS
		keep = box_ops.batched_nms(boxes, scores, filter_inds[1], self.nms_thresh)
		if self.topk_per_image >= 0:
			keep = keep[:self.topk_per_image]

		return {{"pred_boxes": boxes[keep], "scores": scores[keep], "pred_classes": filter_inds[1][keep]}}

	def __call__(self, *params):
		return [self._process_image(*items) for items in zip(*params)]\n"""]


def _generate_FastRCNNHead(global_params, cfg):
	mode = global_params["mode"]

	res = []
	res.append(f"""\n
class FastRCNNHead(nn.Module):

	def __init__(self, in_channels, strides, num_classes, score_thresh={cfg["TEST"]["score_thresh"]}, nms_thresh={cfg["TEST"]["nms_thresh"]}, topk_per_image=100):
		super().__init__()

		self.num_classes = num_classes

		self.box_pooler = RoIPooler({cfg["POOLER"]["resolution"]}, strides, sampling_ratio={cfg["POOLER"]["sampling_ratio"]})
		self.box_head = FastRCNNConvFCHead(in_channels, {cfg["POOLER"]["resolution"]}, num_classes, box_dim=4)
		self.transform = BoxTransform(weights={cfg["box_transform_weights"]})\n""")
	if mode == "train":
		res.append(f"""\
		self.proposal_matcher = Matcher(bg_threshold={cfg["TRAIN"]["iou_thresholds"][0]}, fg_threshold={cfg["TRAIN"]["iou_thresholds"][1]}, allow_low_quality_matches=False)
		self.subsampler = Subsampler(num_samples={cfg["TRAIN"]["batch_size_per_image"]}, positive_fraction={cfg["TRAIN"]["positive_fraction"]}, bg_label=num_classes)\n""")

	res.append(f"""\
		self.find_top_predictions = SelectRCNNPredictions(score_thresh, nms_thresh, topk_per_image)\n""")

	if mode == "train":
		res.append(f"""
		self.loss_weight = {{"roi_cls": {round(cfg["LOSS"]["global_weight"] * (1-cfg["LOSS"]["box_reg_weight"]), 7)}, "roi_loc": {round(cfg["LOSS"]["global_weight"] * cfg["LOSS"]["box_reg_weight"], 7)}}}\n""")

	res.append(f"""
	def _concat_proposals(self, proposals):
		\"\"\"Prepare proposals for pooling. Concatenate it with batch index\"\"\"
		dtype, device = proposals[0].dtype, proposals[0].device
		batch_idx = [torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(proposals)]
		proposals = [torch.cat(item, dim=1) for item in zip(batch_idx, proposals)]
		return torch.cat(proposals)  # (idx, x0, y0, x1, y1)\n""")

	if mode == "train":
		res.append("""
	@torch.no_grad()
	def label_and_sample_proposals(self, proposals, targets):
		\"\"\"
		Каждому proposals подобрать наиболее подходящий target.
		\"\"\"""")

		if cfg["TRAIN"]["append_gt_to_proposal"]:
			res.append("""
		proposals = [torch.cat((g["boxes"], p)) for g, p in zip(targets, proposals)]\n""")

		res.append(f"""
		sampled_input = []
		sampled_target = []
		for proposals_i, targets_i in zip(proposals, targets):
			has_gt = len(targets_i["boxes"]) > 0

			if has_gt:
				# Каждому proposals будет присвоен номер targets и метка
				matched_idxs, matched_labels = self.proposal_matcher(targets_i["boxes"], proposals_i)

				gt_classes = targets_i["classes"][matched_idxs]
				gt_classes.masked_fill_(matched_labels == 0, self.num_classes)
				gt_classes.masked_fill_(matched_labels == -1, -1)
			else:
				gt_classes = proposals_i.new_full((len(proposals_i),), self.num_classes, dtype=torch.long)

			sampled_idxs = torch.cat(self.subsampler(gt_classes))  # Выбираем случайные семплы
			sampled_input.append(proposals_i[sampled_idxs])

			sampled_target_i = {{"classes": gt_classes[sampled_idxs]}}

			if has_gt:
				sampled_targets = matched_idxs[sampled_idxs]

				for k, v in targets_i.items():  # copy all available targets
					if k in sampled_target_i: continue
					sampled_target_i[k] = v[sampled_targets]
			else:
				sampled_target_i["boxes"] = targets_i["boxes"].new_zeros((len(sampled_idxs), 4))

			sampled_target.append(sampled_target_i)

		return sampled_input, sampled_target\n""")

	res.append("""
	def inference(self, proposals, predictions, image_sizes, num_prop_per_image):
		scores_logits, boxes_deltas = predictions

		scores = F.softmax(scores_logits, dim=-1).split(num_prop_per_image)
		proposals = Boxes.xyxy2xywh(proposals[:, 1:]).unsqueeze(1)
		boxes = self.transform.apply_deltas(proposals, boxes_deltas).split(num_prop_per_image)
		return self.find_top_predictions(scores, boxes, image_sizes)\n""")

	if mode == "train":
		res.append(f"""
	def losses(self, proposals, predictions, targets):
		scores_logits, boxes_deltas = predictions

		gt_classes = torch.cat([i["classes"] for i in targets])
		gt_boxes = torch.cat([i["boxes"] for i in targets])
		num_targets = len(gt_classes)

		losses = {{}}
		losses["roi_cls"] = F.cross_entropy(scores_logits, gt_classes, reduction="mean")

		# select only foreground boxes
		fg_inds = (gt_classes < self.num_classes).nonzero(as_tuple=True)[0]
		proposals = Boxes.xyxy2xywh(proposals[fg_inds, 1:])
		gt_boxes = gt_boxes[fg_inds]
		boxes_deltas = boxes_deltas[fg_inds, {"0" if cfg["is_agnostic"] else "gt_classes[fg_inds]"}]\n""")

		if cfg["LOSS"]["bbox_reg_loss_type"] == "smooth_l1":
			res.append(f"""
		losses["roi_loc"] = smooth_l1_loss(
			boxes_deltas, self.transform.get_deltas(proposals, gt_boxes),
			beta={cfg["LOSS"]["smooth_l1_beta"]}, reduction="sum") / num_targets\n""")
		elif cfg["LOSS"]["bbox_reg_loss_type"] == "giou":
			res.append("""
		losses["roi_loc"] = giou_loss(
			self.transform.apply_deltas(proposals, boxes_deltas), gt_boxes,
			reduction="sum") / num_targets\n""")

		res.append("""
		return {k: v * self.loss_weight[k] for k, v in losses.items()}\n""")

	res.append(f"""
	def forward(self, features, image_sizes, proposals{"" if mode == "test" else ", targets=None"}):
		\"\"\"
		Args:
			features (list[tensor]): Список слоев с фичами. Каждый следующий слой
				в 2 раза меньше предыдущего
			image_sizes(list[tuple]): Реальные размеры исходных изображений
			proposals (list[tensor]): Предсказанные обрасти для каждой картинки, xyxy
		\"\"\"\n""")

	if mode == "train":
		res.append("""\
		if targets is not None:
			proposals, targets = self.label_and_sample_proposals(proposals, targets)\n\n""")

	res.append(f"""\
		num_prop_per_image = [len(p) for p in proposals]
		proposals = self._concat_proposals(proposals)  # (M, 5)
		features = self.box_pooler(features, proposals)
		predictions = self.box_head(features)\n""")

	if mode == "train":
		res.append("""
		if targets is None:
			return self.inference(proposals, predictions, image_sizes, num_prop_per_image), {}
		else:
			return proposals, self.losses(proposals, predictions, targets)\n""")
	else:
		res.append("""
		return self.inference(proposals, predictions, image_sizes, num_prop_per_image)\n""")
	return res