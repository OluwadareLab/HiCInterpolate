from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=16, skip_channels=32, out_channels=32):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        combined_channels = out_channels + skip_channels
        self.comb = nn.Sequential(
            nn.Conv2d(
                in_channels=combined_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ), nn.BatchNorm2d(out_channels), nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ), nn.LeakyReLU()
        )

    def forward(self, deep_ftr, skip_connection):
        x = self.upsample(deep_ftr)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.comb(x)
        return x


class FeatureDecoder(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], out_channels=1):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = list(feature_channels)

        self.level1 = DecoderBlock(
            in_channels=self.feature_channels[1],
            skip_channels=self.feature_channels[0],
            out_channels=self.feature_channels[0],
        )
        self.level2 = DecoderBlock(
            in_channels=self.feature_channels[2],
            skip_channels=self.feature_channels[1],
            out_channels=self.feature_channels[1],
        )
        self.level3 = DecoderBlock(
            in_channels=self.feature_channels[3],
            skip_channels=self.feature_channels[2],
            out_channels=self.feature_channels[2],
        )
        self.level4 = DecoderBlock(
            in_channels=self.feature_channels[4],
            skip_channels=self.feature_channels[3],
            out_channels=self.feature_channels[3],
        )

    def forward(self, ftr_stk: List[Tensor]) -> Tensor:
        out = self.level4(ftr_stk[4], ftr_stk[3])
        out = self.level3(out, ftr_stk[2])
        out = self.level2(out, ftr_stk[1])
        out = self.level1(out, ftr_stk[0])
        return out
