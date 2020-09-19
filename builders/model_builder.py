import os
import zlib, base64

FOLDER = "."

def build_model(cfg, test_only=False, engine="pytorch"):
	os.makedirs(FOLDER, exist_ok=True)

	code, libs = generate_model(cfg, test_only)

	with open(os.path.join(FOLDER, "model.py"), "w") as f:
		f.write(code)

	return libs


def generate_model(cfg, test_only=False, engine="pytorch"):
	assert isinstance(test_only, bool)
	assert isinstance(engine, str)

	assert cfg["BACKBONE"]["name"] in ["fpn_resnet"]

	if engine == "pytorch":
		return generate_model_pytorch(cfg, test_only)
	
	raise NotImplementedError(f"Unimplemented engine {engine}")


def generate_model_pytorch(cfg, test_only):
	res = []
	libs = set()
	res.append("###  Automatically-generated file  ###\n\n")

	def generate_imports():
		res.append("""\
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import ResNet, FPN, RPN, FastRCNNHead
from .utils import Images\n""")
		libs.add("parts/resnet.py")
		libs.add("parts/fpn.py")
		libs.add("parts/rpn.py")
		libs.add("parts/fast_rcnn.py")
		libs.add("utils/images.py")

		if not test_only:
			res.append(f"from .layers import Conv2d\n")
			libs.add("layers/conv_wrapper.py")

		res.append("\n")

	def generate_config():
		cfg_str = base64.b64encode(zlib.compress(str(cfg).encode(), 9)).decode()
		cfg_str = [cfg_str[i:i+77] for i in range(0, len(cfg_str), 77)]
		assert zlib.decompress(base64.b64decode("".join(cfg_str).encode())).decode() == str(cfg)
		cfg_str = "# " + "\n# ".join(cfg_str)

		res.append("# Configuration (base64, zlib):\n")
		res.append(cfg_str)
		res.append("\n\n")

	def generate_Model():
		res.append(f"""\
class FasterRCNN(nn.Module):
	\"\"\"Реализация FasterRCNN для PyTorch.\"\"\"

	def __init__(self, in_channels, num_classes, score_thresh={cfg["ROI_HEAD"]["StandardROIHeads"]["TEST"]["score_thresh"]}, nms_thresh={cfg["ROI_HEAD"]["StandardROIHeads"]["TEST"]["nms_thresh"]}, topk_per_image=100):
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
		bottom_up = ResNet(in_channels=in_channels, out_features={cfg["BACKBONE"]["RESNETS"]["out_features"]})
		self.backbone = FPN(bottom_up, out_channels={cfg["BACKBONE"]["FPN"]["out_channels"]})
		self.proposal_generator = RPN(self.backbone.out_channels[0], self.backbone.out_strides)
		self.roi_heads = FastRCNNHead(self.backbone.out_channels[0], self.backbone.out_strides,
			num_classes, score_thresh, nms_thresh, topk_per_image)

	def _assert_inputs(self, images{"" if test_only else ", targets"}):
		assert len(images) > 0
		assert isinstance(images, list) and len(images) > 0
		assert all([i.dim() == 3 and i.size(0) == self.in_channels for i in images])\n""")

		if not test_only:
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
	def forward(self, images{"" if test_only else ", targets=None"}):
		\"\"\"
		Args:
			images (list[tensor]): лист, содержащий картинки в виде тензора с
				размером (in_channels, Hi, Wi)
		\"\"\"""")
		if test_only:
			res.append("""
		assert not self.training, \"This model is only for evaluation!\"""")

		res.append(f"""
		self._assert_inputs(images{"" if test_only else ", targets"})

		images, image_sizes = Images.concat_images(images, self.backbone.size_divisibility)
		features = self.backbone(images)
		proposals{"" if test_only else ", rpn_loss"} = self.proposal_generator(features, image_sizes{"" if test_only else ", targets"})
		predicts{"" if test_only else ", roi_loss"} = self.roi_heads(features, image_sizes, [i[1] for i in proposals]{"" if test_only else ", targets"})
		return predicts{"" if test_only else ", {**rpn_loss, **roi_loss}"}\n""")

		if not test_only:
			res.append("""
	def extract(self):
		\"\"\"
		Подготовить модель для тестирования, заменить все Conv2d на
		nn.Conv2d, объединив их с BatchNorm
		\"\"\"
		return Conv2d.extract_all(self)\n""")

	generate_imports()
	generate_config()
	generate_Model()

	return "".join(res), libs