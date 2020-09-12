import torch
from build.model import Model

net = Model(in_channels=3, num_classes=5)
net(torch.rand(1, 3, 512, 512), [(512, 512)], 1)
net(torch.rand(1, 3, 512, 512), [(512, 512)], 1)

torch.save(net.state_dict(), "weight.pt")
torch.save(torch.rand(1, 3, 512, 512), "data.pt")