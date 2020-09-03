import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

class Model(nn.Module):

	def __init__(self, backbone):
		super().__init__()
		self.backbone = backbone

	def forward(self, x):
		return self.backbone(x)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85

net = Model(build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3)))

temp = torch.load("weight.pt")
net.load_state_dict({k: temp[k] for k in net.state_dict()})
net.eval()

with torch.no_grad():
	torch.save(net(torch.load("data.pt")), "res1.pt")