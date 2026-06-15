from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SkipFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels*4, channels*4,
                      kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels*4, channels*4, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x0: Tensor, x2: Tensor) -> Tensor:
        alpha = self.gate(torch.cat([x0, x2], dim=1))
        return self.refine(alpha * x0 + (1.0 - alpha) * x2)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        upsampled = self.up_sample(x)
        fused = torch.cat([upsampled, skip], dim=1)
        refined = self.refine(fused)
        return refined


class FeatureDecoder(nn.Module):
    def __init__(self, feature_channels: List[int] = [256, 128, 64, 32]):
        super().__init__()
        self.decoder_head = nn.ModuleList([
            DecoderBlock(feature_channels[idx+1], feature_channels[idx])
            for idx in range(len(feature_channels)-1)
        ])

    def forward(self, xs: List[Tensor]) -> Tensor:
        for idx in reversed(range(len(self.decoder_head))):
            x, skip = xs[idx+1], xs[idx]
            x = self.decoder_head[idx](x, skip)
        return x
