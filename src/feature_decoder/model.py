from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SkipFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, skip0: Tensor, skip2: Tensor) -> Tensor:
        alpha = self.gate(torch.cat([skip0, skip2], dim=1))
        return self.refine(alpha * skip0 + (1.0 - alpha) * skip2)


class DecoderBlock(nn.Module):
    def __init__(self, skip_channels: int, out_channels: int, kernel_size: int, deep_channels: int = 0):
        super().__init__()
        self.skip_fusion = SkipFusion(skip_channels)
        self.deep_proj = None
        total_channels = skip_channels * 4
        if deep_channels > 0:
            self.deep_proj = nn.Conv2d(
                deep_channels, skip_channels, kernel_size=1, bias=False)
            total_channels += skip_channels

        padding = kernel_size // 2
        self.refine = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, interp: Tensor, warp0: Tensor, warp2: Tensor,
                skip0: Tensor, skip2: Tensor, deep: Tensor = None) -> Tensor:
        fused_skip = self.skip_fusion(skip0, skip2)
        parts = [fused_skip, interp, warp0, warp2]
        if self.deep_proj is not None:
            deep = F.interpolate(deep, size=interp.shape[-2:], mode="nearest")
            parts.insert(0, self.deep_proj(deep))
        return self.refine(torch.cat(parts, dim=1))


class FeatureDecoder(nn.Module):
    def __init__(self, cfg, feature_channels: List[int] = None, out_channels: int = 256):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = feature_channels or [256, 128, 64, 32, 16]

        self.level5 = DecoderBlock(
            self.feature_channels[4], 16, kernel_size=7, deep_channels=0)
        self.level4 = DecoderBlock(
            self.feature_channels[3], 32, kernel_size=5, deep_channels=16)
        self.level3 = DecoderBlock(
            self.feature_channels[2], 64, kernel_size=3, deep_channels=32)
        self.level2 = DecoderBlock(
            self.feature_channels[1], 128, kernel_size=3, deep_channels=64)
        self.level1 = DecoderBlock(
            self.feature_channels[0], out_channels, kernel_size=1, deep_channels=128)

    def forward(self, interpolations: List[Tensor], warps0: List[Tensor], warps2: List[Tensor], skips0: List[Tensor], skips2: List[Tensor]) -> Tensor:
        out = self.level5(
            interpolations[4], warps0[4], warps2[4], skips0[4], skips2[4])
        out = self.level4(
            interpolations[3], warps0[3], warps2[3], skips0[3], skips2[3], out)
        out = self.level3(
            interpolations[2], warps0[2], warps2[2], skips0[2], skips2[2], out)
        out = self.level2(
            interpolations[1], warps0[1], warps2[1], skips0[1], skips2[1], out)
        out = self.level1(
            interpolations[0], warps0[0], warps2[0], skips0[0], skips2[0], out)
        return out
