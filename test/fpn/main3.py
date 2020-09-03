import torch
from build.model import Model

net = Model(in_channels=3, num_classes=5)
net.load_state_dict(torch.load("weight_res.pt"))
net.eval()

with torch.no_grad():
	torch.save(net(torch.load("data.pt")), "res3.pt")