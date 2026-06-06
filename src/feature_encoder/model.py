from typing import List
import torch.nn as nn
from torch import Tensor


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = self._dilated_padding(kernel_size)
        self.down_ftr = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=2,
            ),
            nn.LeakyReLU(),
        )

    @staticmethod
    def _dilated_padding(kernel_size: int, dilation: int = 2) -> int:
        return (kernel_size - 1) * dilation // 2

    def forward(self, x: Tensor) -> Tensor:
        x = self.down_ftr(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, cfg, in_channels=1, out_channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.cfg = cfg

        self.stages = nn.ModuleList([
            EncoderBlock(
                out_channels[i], out_channels[i + 1], self._kernel_size(i),
            )
            for i in range(len(out_channels) - 1)
        ])

    @staticmethod
    def _kernel_size(stage_idx: int) -> int:
        if stage_idx < 2:
            return 7
        if stage_idx == 2:
            return 5
        return 3

    def forward(self, ftr: Tensor) -> List[Tensor]:
        outputs = []
        x = ftr
        outputs.append(x)
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs
