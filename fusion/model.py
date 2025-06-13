from typing import List
from torch.nn import Module, Conv2d, LeakyReLU, ReLU, functional as F, ModuleList
from torch import Tensor
import torch
import config


class Block(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channels,
                            out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.relu1 = LeakyReLU(negative_slope=0.2)
        self.conv2 = Conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.relu2 = LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        output = self.relu2(x)

        return output


class Fusion(Module):
    def __init__(self):
        super().__init__()
        self.convs = ModuleList()
        self.levels = config.FUSION_PYRAMID_LEVEL
        prev_out_channels = 0
        for i in range(config.FUSION_PYRAMID_LEVEL-1):
            m = config.UNIQUE_LEVELS
            k = config.INIT_OUT_CHANNELS
            out_channels = (k << i) if i < m else (k << m)
            in_channels = (prev_out_channels + out_channels + 3) * 2 + 4

            convs = ModuleList()
            convs.append(Conv2d(in_channels=out_channels*2 if i < m else in_channels,
                                out_channels=out_channels, kernel_size=2, padding='same'))
            convs.append(Block(in_channels=in_channels + out_channels,
                         out_channels=out_channels, kernel_size=3))
            self.convs.append(convs)
            prev_out_channels = prev_out_channels + out_channels

        self.output_conv = Conv2d(
            in_channels=config.INIT_OUT_CHANNELS, out_channels=3, kernel_size=1)

    def forward(self, pyramid: List[Tensor]) -> Tensor:
        if len(pyramid) != self.levels:
            raise ValueError(
                '[ERROR] fusion called with different number of pyramid levels ' f'{len(pyramid)} than it was configured for, {self.levels}.')
        net = pyramid[-1]
        for i in reversed(range(0, self.levels-1)):
            level_size = (pyramid[i].shape)[2:4]
            net = F.interpolate(net, size=level_size, mode='nearest')
            net = self.convs[i][0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = self.convs[i][1](net)
        net = self.output_conv(net)

        return net
