from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.refine(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.dec = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        up = self.up(x)
        fused = torch.cat([up, skip], dim=1)
        return self.dec(fused)


class FeatureDecoder(nn.Module):
    def __init__(self, base_channels: int = 32, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        kernel = 3
        dec_blocks = []
        prev_channels = base_channels
        for idx in range(depth):
            dec_blocks.append(DecoderBlock(
                base_channels * (2 ** (idx + 1)), prev_channels, dropout=dropout
            ))
            prev_channels = base_channels * (2 ** (idx + 1))
        self.dec_blocks = nn.ModuleList(dec_blocks)

    def forward(self, ftrs: List[Tensor]) -> Tensor:
        x = ftrs[-1]
        for idx in reversed(range(len(self.dec_blocks))):
            skip = ftrs[idx]
            x = self.dec_blocks[idx](x, skip)
        return x
