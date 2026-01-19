from torch.nn import Module, Conv2d, ReLU, ModuleList
from torch import Tensor
import torch


class Block(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels,
                           out_channels=out_channels, kernel_size=kernel_size, padding="same")
        self.relu = ReLU()

    def forward(self, input):
        x = input
        x = self.conv(x)
        output = self.relu(x)

        return output


class FlowEstimator(Module):
    def __init__(self, num_of_convs: int, in_channels: int, out_channels: int):
        super().__init__()
        self.convs = ModuleList()
        for i in range(num_of_convs):
            self.convs.append(Block(in_channels=in_channels,
                                    out_channels=out_channels, kernel_size=3))
            in_channels = out_channels

        out_channels = (int)(out_channels/2)
        self.convs.append(Block(in_channels=in_channels,
                                out_channels=out_channels, kernel_size=1))

        in_channels = out_channels
        self.convs.append(Conv2d(in_channels=in_channels,
                                 out_channels=2, kernel_size=1, padding="same"))

    def forward(self, feature_a: Tensor, feature_b: Tensor) -> Tensor:
        net = torch.cat([feature_a, feature_b], dim=1)
        for conv in self.convs:
            net = conv(net)
        return net
