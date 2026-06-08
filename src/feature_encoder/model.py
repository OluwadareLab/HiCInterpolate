from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_maxpool: bool = True):
        super().__init__()
        hidden_ch = out_channels // 2

        self.fine_path = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.structural_path = nn.Sequential(
            # Using dilation=2 on a 3x3 kernel gives a 5x5 field without the heavy blur
            nn.Conv2d(in_channels, hidden_ch, kernel_size=3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Identity mapping to allow a local residual shortcut if channel sizes match
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2) if use_maxpool else nn.Identity()

    def forward(self, x):
        # 1. Extract parallel features
        f1 = self.fine_path(x)
        f2 = self.structural_path(x)
        combined = torch.cat([f1, f2], dim=1)

        # 2. Local residual connection to maintain crisp features
        skip_out = combined + self.shortcut(x)

        # 3. Downsample for the next level
        pooled_out = self.pool(skip_out)

        # We return pooled_out for the next deep layer,
        # and skip_out to send across to the UNet Decoder
        return pooled_out, skip_out


class FeatureEncoder(nn.Module):
    def __init__(self, cfg, in_channels=48, out_channels=[32, 64, 128, 256]):
        super().__init__()
        self.cfg = cfg

        self.level1 = EncoderBlock(
            in_channels, out_channels[0], use_maxpool=True)
        self.level2 = EncoderBlock(
            out_channels[0], out_channels[1], use_maxpool=True)
        self.level3 = EncoderBlock(
            out_channels[1], out_channels[2], use_maxpool=True)
        self.level4 = EncoderBlock(
            out_channels[2], out_channels[3], use_maxpool=False)

    def forward(self, ftr: Tensor) -> List[Tensor]:
        outputs = []
        x, out1 = self.level1(ftr)
        outputs.append(out1)
        x, out2 = self.level2(x)
        outputs.append(out2)
        x, out3 = self.level3(x)
        outputs.append(out3)
        x, out4 = self.level4(x)
        outputs.append(out4)
        return outputs
