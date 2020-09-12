import torch
from build.model import Model

net = Model(in_channels=3, num_classes=5)

temp = torch.load("weight.pt")
print("Problems with:\n" + "\n".join([k for k in net.state_dict() if k not in temp]))
net.load_state_dict({k: temp.get(k, v) for k, v in net.state_dict().items()})
net.eval()

data = torch.load("data.pt")

with torch.no_grad():
	torch.save(net(data, [data.shape[-2:]]), "res2_1.pt")
	net.extract()
	torch.save(net(data, [data.shape[-2:]]), "res2_2.pt")

	torch.save(net.state_dict(), "weight_res.pt")