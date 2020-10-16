import time
import torch
from torch import optim

from model import FasterRCNN

net = FasterRCNN(in_channels=3, num_classes=5).cuda()

net.load_state_dict(torch.load("weight.pt"))
net.backbone.bottom_up.freeze(2)
optimizer = optim.Adam(net.parameters())

data = [torch.load("data.pt").cuda()]
targets = [{k: v.cuda() for k, v in torch.load("targets.pt").items()}]

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

#with torch.no_grad():
for i in range(3):
	torch.cuda.synchronize()
	t = time.time()
	res, losses = net(data, targets)
	torch.cuda.synchronize()
	print(time.time() - t)
	print(losses)
	# losses = sum(losses.values())
	losses = losses["rpn_cls"]
	optimizer.zero_grad()
	losses.backward()
	optimizer.step()

	#net.extract()
	#net.eval()
	#torch.save(net(data, [data.shape[-2:]], targets), "res2_2.pt")

	#torch.save(net.state_dict(), "weight_res.pt")