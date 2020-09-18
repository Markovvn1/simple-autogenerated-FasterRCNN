import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.events import EventStorage
from detectron2.structures import Instances, Boxes
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5
cfg.MODEL.RPN.SMOOTH_L1_BETA = 1.0
cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

net = build_model(cfg)

state_dict_map = {
	"roi_heads.box_head.cls_score.weight": "roi_heads.box_predictor.cls_score.weight",
	"roi_heads.box_head.cls_score.bias": "roi_heads.box_predictor.cls_score.bias",
	"roi_heads.box_head.bbox_pred.weight": "roi_heads.box_predictor.bbox_pred.weight",
	"roi_heads.box_head.bbox_pred.bias": "roi_heads.box_predictor.bbox_pred.bias",
}

temp = torch.load("weight.pt")
temp = {state_dict_map.get(k, k): v for k, v in temp.items()}
print("Problems with:\n" + "\n".join([k for k in net.state_dict() if k not in temp]))
net.load_state_dict({k: temp.get(k, v) for k, v in net.state_dict().items()})
net.eval()

#targets = [Instances((512, 512))]
#targets[0].gt_boxes = Boxes(torch.load("targets.pt"))

data = [{"image": torch.load("data.pt").cuda()}]

storage_4del = EventStorage(0).__enter__()

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

with torch.no_grad():
	torch.save(net(data), "res1.pt")

storage_4del.__exit__(None, None, None)