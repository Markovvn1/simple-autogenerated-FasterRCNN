import torch
from build.model import FasterRCNN

net = FasterRCNN(in_channels=3, num_classes=5)
net([torch.rand(1, 3, 512, 512)])
net([torch.rand(1, 3, 512, 512)])

torch.save(net.state_dict(), "weight.pt")
torch.save(torch.rand(1, 3, 512, 512), "data.pt")

a = torch.rand(10, 4) * 256
a[..., 2:] += a[..., :2]
torch.save(a, "targets.pt")