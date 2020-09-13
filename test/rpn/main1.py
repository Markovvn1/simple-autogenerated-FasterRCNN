import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.events import EventStorage
from detectron2.structures import Instances, Boxes
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.proposal_generator.build import build_proposal_generator

class Model(nn.Module):

	def __init__(self, cfg):
		super().__init__()
		self.backbone = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
		self.rpn = build_proposal_generator(cfg, self.backbone.output_shape())

	def forward(self, x, targets):
		features = self.backbone(x)
		return self.rpn(ImageListPlug([x.shape[-2:]]), features, targets)

class ImageListPlug:
	def __init__(self, image_sizes):
		self.image_sizes = image_sizes


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5
cfg.MODEL.RPN.SMOOTH_L1_BETA = 1.0

net = Model(cfg)

temp = torch.load("weight.pt")
print("Problems with:\n" + "\n".join([k for k in net.state_dict() if k not in temp]))
net.load_state_dict({k: temp.get(k, v) for k, v in net.state_dict().items()})
#net.eval()

targets = [Instances((512, 512))]
targets[0].gt_boxes = Boxes(torch.load("targets.pt"))

storage_4del = EventStorage(0).__enter__()

with torch.no_grad():
	torch.save(net(torch.load("data.pt"), targets), "res1.pt")

storage_4del.__exit__(None, None, None)