import torch
from torch.nn import functional as F

from resnet.resnet import ResNet

USE_GPU = True
data_type = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def shape_test():
    N, C, H, W = 10, 3, 32, 32
    x = torch.randn((N, C, H, W), dtype=data_type, device=device)
    model = ResNet(3)
    model.to(device=device, dtype=data_type)
    out = model(x)
    y = torch.randint(10, size=(N, ), device=device, dtype=torch.int64)
    loss = F.cross_entropy(out, y)
    print(loss)


if __name__ == '__main__':
    shape_test()
