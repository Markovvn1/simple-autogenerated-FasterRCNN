import torch
from model import FasterRCNN

net = FasterRCNN(in_channels=3, num_classes=5).cuda()
net([torch.rand(3, 512, 512).cuda()])
net([torch.rand(3, 512, 512).cuda()])

torch.save(net.state_dict(), "weight.pt")
torch.save(torch.rand(3, 512, 512), "data.pt")

a = torch.rand(10, 4) * 256
a[..., 2:] += a[..., :2]
torch.save({"boxes": a, "classes": (torch.rand(10) * 5).floor().long()}, "targets.pt")