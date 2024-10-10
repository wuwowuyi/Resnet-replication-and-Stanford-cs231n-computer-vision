import torch
from torch import nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False, *, downsampling="zero_padding", prenorm=False):
        super().__init__()

        self.prenorm = prenorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsampling = downsampling if in_channels != out_channels else None
        if self.downsampling == 'zero_padding':
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def _zero_padding(self, x):
        """Option A described in paper. """
        out = self.maxpool(x)  # out.shape=(N, in_channels, H//2, W//2)
        padding = torch.zeros_like(out, device=out.device, dtype=out.dtype)  # suppose out_channels = 2 * in_channels
        out = torch.cat((out, padding), dim=1)
        out = self.bn3(out)
        return out

    def forward(self, x):
        if self.prenorm:
            out = self.bn1(self.relu(self.conv1(x)))
            out = self.bn2(self.relu(self.conv2(out)))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

        if self.downsampling == 'zero_padding':
            x = self._zero_padding(x)

        if self.prenorm:
            return out + x
        else:
            out = self.relu(out + x)
            return out


class ResNet(nn.Module):
    """ResNet for CIFAR1-10.
    Design follows section 4.2 of the original ResNet paper.
    """

    def __init__(self, n, prenorm=False):
        """
        n = 3 is 20 layers
        """
        super().__init__()

        self.prenorm = prenorm

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # blocks1 input: 16 * 32 * 32, output 16 * 32 * 32
        # blocks2 output: 32 * 16 * 16
        # blocks3 output: 64 * 8 * 8
        self.blocks1 = nn.Sequential(
            *[Block(16, 16, prenorm=prenorm) for i in range(n)],
        )
        self.blocks2 = nn.Sequential(
            Block(16, 32, stride=2, prenorm=prenorm),
            *[Block(32, 32, prenorm=prenorm) for i in range(n - 1)],
        )
        self.blocks3 = nn.Sequential(
            Block(32, 64, stride=2, prenorm=prenorm),
            *[Block(64, 64, prenorm=prenorm) for i in range(n - 1)],
        )

        self.pool = nn.AvgPool2d(8, 1)  # output 1 * 1 spatial size
        self.fc = nn.Linear(64, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters is {total_params}")

    def forward(self, x):
        if self.prenorm:
            x = self.bn1(self.relu(self.conv1(x)))
        else:
            x = self.relu(self.bn1(self.conv1(x)))

        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.pool(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x








