import torch
from build.resnet import ResNet

net = ResNet(in_channels=3, out_features=["res2", "res3", "res4", "res5"])
# print(net)
# print(net.state_dict().keys())
net.load_state_dict(torch.load("weight_res.pt"))
net.eval()

with torch.no_grad():
	torch.save(net(torch.load("data.pt")), "res3.pt")