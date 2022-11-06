import torch
import torch.nn as nn

architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    [(1, 64, 1, 0),
    (3, 64, 1, 1),
    (1, 256, 1, 0), 3]

]

class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BuildingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

