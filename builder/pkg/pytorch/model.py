from pkg.module_builder import ModuleBuilderBase


class ModuleBuilder(ModuleBuilderBase):

	def __init__(self):
		super().__init__({"model.FasterRCNN"})

	def _assert_cfg(self, module, cfg):
		if module == "model.FasterRCNN":
			assert cfg["BACKBONE"]["name"] in ["RESNET"]
			assert cfg["NECK"]["name"] in ["FPN"]
			assert cfg["PROPOSAL_GENERATOR"]["name"] in ["RPN"]
			assert cfg["ROI_HEADS"]["name"] in ["STANDARD"]

	def _dependencies(self, module, global_params, cfg):
		if module == "model.FasterRCNN":
			res = {"images.concat_images": None}
			if cfg["BACKBONE"]["name"] == "RESNET":
				res["resnet.ResNet"] = cfg["BACKBONE"]["RESNET"]
			if cfg["NECK"]["name"] == "FPN":
				res["fpn.FPN"] = cfg["NECK"]["FPN"]
			if cfg["PROPOSAL_GENERATOR"]["name"] == "RPN":
				res["rpn.RPN"] = cfg["PROPOSAL_GENERATOR"]["RPN"]
			if cfg["ROI_HEADS"]["name"] == "STANDARD":
				res["fast_rcnn.FastRCNNHead"] = cfg["ROI_HEADS"]["STANDARD"]
			if global_params["mode"] == "train":
				res["conv_wrapper.Conv2d"] = None
			return res

	def _init_file(self, dep):
		if "model.FasterRCNN" in dep: return "from .model import FasterRCNN"
		return None

	def _generate(self, global_params, dep, childs):
		res = []
		res.append("""\
import torch
import torch.nn as nn\n""")

		first = True
		parts = {"resnet.ResNet", "fpn.FPN", "rpn.RPN", "fast_rcnn.FastRCNNHead"} & childs.keys()
		if parts:
			if first: res.append("\n")
			first = False
			res.append("from .parts import " + ", ".join([i[i.find(".")+1:] for i in parts]) + "\n")

		utils = {} & childs.keys()
		if [1 for i in childs.keys() if i[:i.find(".")] == "images"]: utils.add("images.Images")
		if utils:
			if first: res.append("\n")
			first = False
			res.append("from .utils import " + ", ".join([i[i.find(".")+1:] for i in utils]) + "\n")

		layers = {"conv_wrapper.Conv2d"} & childs.keys()
		if layers:
			if first: res.append("\n")
			first = False
			res.append("from .layers import " + ", ".join([i[i.find(".")+1:] for i in layers]) + "\n")

		if "model.FasterRCNN" in dep:
			res.extend(_generate_FasterRCNN(global_params, dep["model.FasterRCNN"]))

		return "".join(res)


def _generate_FasterRCNN(global_params, cfg):
	mode = global_params["mode"]

	res = []
	res.append(f"""\n
class FasterRCNN(nn.Module):
	\"\"\"Реализация FasterRCNN для PyTorch.\"\"\"

	def __init__(self, in_channels, num_classes, \
score_thresh={cfg["ROI_HEADS"][cfg["ROI_HEADS"]["name"]]["TEST"]["score_thresh"]}, \
nms_thresh={cfg["ROI_HEADS"][cfg["ROI_HEADS"]["name"]]["TEST"]["nms_thresh"]}, topk_per_image=100):
		\"\"\"
		Args:
			in_channels (int): Количество каналов во входном изображении
			num_channels (int): Количество классов, для классификации
			score_thresh (float): Результаты, имеющие меньший score будут отсеяны
			nms_thresh (float): Допустимый показатель IOU для результатов
			topk_per_image (int): Максимально допустимое количество результатов.
				Если < 0, то будут возвращены все результаты
		\"\"\"
		super().__init__()
		self.in_channels = in_channels
		bottom_up = ResNet(in_channels=in_channels, out_features={cfg["BACKBONE"][cfg["BACKBONE"]["name"]]["out_features"]})
		self.backbone = FPN(bottom_up, out_channels={cfg["NECK"][cfg["NECK"]["name"]]["out_channels"]})
		self.proposal_generator = RPN(self.backbone.out_channels[0], self.backbone.out_strides)
		self.roi_heads = FastRCNNHead(self.backbone.out_channels[0], self.backbone.out_strides,
			num_classes, score_thresh, nms_thresh, topk_per_image)

	def _assert_inputs(self, images{"" if mode == "test" else ", targets"}):
		if isinstance(images, torch.Tensor):
			assert images.dim() == 4 and images.size(1) == self.in_channels
		else:
			assert isinstance(images, list) and len(images) > 0
			assert all([i.dim() == 3 and i.size(0) == self.in_channels for i in images])\n""")

	if mode == "train":
		res.append(f"""
		if targets is None: return
		assert isinstance(targets, (list, tuple)) and len(targets) == len(images)
		for item in targets:
			pre_len = -1
			for k, v in item.items():
				assert isinstance(v, torch.Tensor)
				assert pre_len == -1 or len(v) == pre_len, "Некоторые таргеты имеют разные размеры"
				pre_len = len(v)\n""")

	res.append(f"""
	def forward(self, images{"" if mode == "test" else ", targets=None"}):
		\"\"\"
		Args:
			images (list[tensor]): лист, содержащий картинки в виде тензора с
				размером (in_channels, Hi, Wi)
		\"\"\"""")
	if mode == "test":
		res.append("""
		assert not self.training, \"This model is only for evaluation!\"""")

	res.append(f"""
		self._assert_inputs(images{"" if mode == "test" else ", targets"})

		if isinstance(images, torch.Tensor):
			image_sizes = [images.shape[-2:]] * len(images)
		else:
			images, image_sizes = Images.concat_images(images, self.backbone.size_divisibility)
		features = self.backbone(images)
		proposals{"" if mode == "test" else ", rpn_loss"} = self.proposal_generator(features, image_sizes{"" if mode == "test" else ", targets"})
		predicts{"" if mode == "test" else ", roi_loss"} = self.roi_heads(features, image_sizes, [i[1] for i in proposals]{"" if mode == "test" else ", targets"})
		return predicts{"" if mode == "test" else ", {**rpn_loss, **roi_loss}"}\n""")

	if mode == "train":
		res.append("""
	def extract(self):
		\"\"\"
		Подготовить модель для тестирования, заменить все Conv2d на
		nn.Conv2d, объединив их с BatchNorm
		\"\"\"
		return Conv2d.extract_all(self)\n""")
	return res