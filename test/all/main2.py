import torch
from model import FasterRCNN

net = FasterRCNN(in_channels=3, num_classes=5).cuda()

net.load_state_dict(torch.load("weight.pt"))
#net.eval()

data = [torch.load("data.pt").cuda()]
targets = [{k: v.cuda() for k, v in torch.load("targets.pt").items()}]

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

with torch.no_grad():
	torch.save(net(data, targets), "res2.pt")
	#net.extract()
	#net.eval()
	#torch.save(net(data, [data.shape[-2:]], targets), "res2_2.pt")

	#torch.save(net.state_dict(), "weight_res.pt")