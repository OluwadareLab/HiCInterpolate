from typing import List
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
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
        return self.encoder(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        down = self.downsample(x)
        return self.enc(down)


class FeatureEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, base_channels: int = 32, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        kernel = 3
        self.input_enc = ConvBlock(
            input_channels, base_channels, dropout=dropout)
        enc_blocks = []
        prev_channels = base_channels
        for idx in range(depth):
            enc_blocks.append(EncoderBlock(
                prev_channels, base_channels * (2 ** (idx + 1)), dropout=dropout
            ))
            prev_channels = base_channels * (2 ** (idx + 1))

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.bottleneck = EncoderBlock(
            prev_channels, prev_channels * 2, dropout=dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, List[Tensor]]:
        enc_ftrs = [None] * (len(self.enc_blocks) + 1)
        enc_ftrs[0] = self.input_enc(x)
        for idx, block in enumerate(self.enc_blocks):
            enc_ftrs[idx+1] = block(enc_ftrs[idx])
        return enc_ftrs
