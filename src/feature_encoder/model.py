from typing import List
import torch.nn as nn
from torch import Tensor


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample: bool):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.proj(x)
        return self.downsample(skip), skip


class FeatureEncoder(nn.Module):
    def __init__(self, cfg, in_channels: int = 384, out_channels: List[int] = None):
        super().__init__()
        self.cfg = cfg
        self.out_channels = out_channels or [256, 128, 64, 32, 16]
        kernels = [1, 3, 3, 5, 7]

        blocks = []
        prev_channels = in_channels
        for idx, (channels, kernel) in enumerate(zip(self.out_channels, kernels)):
            blocks.append(EncoderBlock(
                prev_channels, channels, kernel_size=kernel,
                downsample=idx < len(self.out_channels) - 1,
            ))
            prev_channels = channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, ftr: Tensor) -> List[Tensor]:
        outputs = []
        x = ftr
        for block in self.blocks:
            x, skip = block(x)
            outputs.append(skip)
        return outputs
