import torch
from build.parts import ResNet

net = ResNet(in_channels=3, out_features=["res2", "res3", "res4", "res5"])
net.load_state_dict(torch.load("weight.pt"))
net.eval()

with torch.no_grad():
	torch.save(net(torch.load("data.pt")), "res2_1.pt")
	net.extract()
	torch.save(net(torch.load("data.pt")), "res2_2.pt")

	torch.save(net.state_dict(), "weight_res.pt")