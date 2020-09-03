import torch
from build.model import Model

net = Model(in_channels=3, num_classes=5)
net.load_state_dict(torch.load("weight.pt"))
net.eval()

with torch.no_grad():
	torch.save(net(torch.load("data.pt")), "res2_1.pt")
	net.extract()
	torch.save(net(torch.load("data.pt")), "res2_2.pt")

	torch.save(net.state_dict(), "weight_res.pt")