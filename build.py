from builder import Builder


import yaml
with open("configs/faster_rcnn_R_50_FPN.yaml", "r") as f:
	cfg = yaml.safe_load(f)

builder = Builder("pytorch", "build_new")
# builder.build("pooler.RoIPooler", {"mode": "train"}, cfg["FASTER_RCNN"]["ROI_HEADS"]["STANDARD"]["POOLER"])
# builder.build("resnet.ResNet", {"mode": "train"}, cfg["FASTER_RCNN"]["BACKBONE"]["RESNET"])
# builder.build("fpn.FPN", {"mode": "train"}, cfg["FASTER_RCNN"]["NECK"]["FPN"])
builder.build("rpn.RPN", {"mode": "train"}, cfg["FASTER_RCNN"]["PROPOSAL_GENERATOR"]["RPN"])
# builder.build("rpn.SelectRPNProposals", {"mode": "train"}, None)
