from typing import List
import torch
import torch.nn as nn
from torch import Tensor


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
        return self.refine(fused)


class FeatureDecoder(nn.Module):
    def __init__(self, feature_channels: List[int] = None):
        super().__init__()
        feature_channels = feature_channels or [16, 32, 64, 128]
        # One DecoderBlock per adjacent pair: (coarser → finer)
        # decoder_head[i]: upsamples feature_channels[i+1] → feature_channels[i]
        self.decoder_head = nn.ModuleList([
            DecoderBlock(feature_channels[idx]*2, feature_channels[idx])
            for idx in range(len(feature_channels) - 1)
        ])

    def forward(self, xs: List[Tensor]) -> Tensor:
        """Progressive U-Net decode: start from coarsest, upsample to finest.

        xs[0] = finest   (e.g. [B, 16, H,   W  ])
        xs[-1] = coarsest (e.g. [B, 128, H/8, W/8])

        Chain: xs[-1] → decoder_head[-1] (skip=xs[-2]) →
               ...   → decoder_head[0]  (skip=xs[0])  → output
        """
        x = xs[-1]  # start from coarsest interpolated feature
        for idx in reversed(range(len(self.decoder_head))):
            skip = xs[idx]  # finer-scale interpolated feature as skip
            x = self.decoder_head[idx](x, skip)
        return x
