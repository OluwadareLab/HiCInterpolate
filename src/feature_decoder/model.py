import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=16, shared_channels=32, skip_channels=32,  out_channels=32):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        combined_channels = out_channels + shared_channels
        self.conv1 = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        combined_channels = out_channels + skip_channels
        self.conv2 = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, unique_ftr, shared_ftr, skip_connection):
        x = self.upsample(unique_ftr)

        x = torch.cat([x, shared_ftr], dim=1)
        x = self.act(self.bn1(self.conv1(x)))

        x = torch.cat([x, skip_connection], dim=1)
        x = self.act(self.bn2(self.conv2(x)))

        x = self.act(self.bn3(self.conv3(x)))

        return x


class DecoderBlock8To16(nn.Module):
    def __init__(self, in_channels=1024, skip_channels=512, out_channels=512):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=4,
            padding=0,
        )
        combined_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        return x


class DecoderBlock16To32(nn.Module):
    def __init__(self, in_channels=512, skip_channels=256, out_channels=256):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        combined_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        return x


class OutputProjection(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels//2,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1
        )
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels//2,
            out_channels=in_channels//4,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1
        )
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        self.conv3 = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels//8,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1
        )
        self.bn3 = nn.BatchNorm2d(in_channels//8)
        self.conv4 = nn.Conv2d(
            in_channels=in_channels//8,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1
        )
        self.act = nn.LeakyReLU()
        self.final_act = nn.Softplus()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.act(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out = self.final_act(out)
        return out


class FeatureDecoder(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], out_channels=1):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = [x * 2 for x in feature_channels]
        self.output_proj = OutputProjection(
            in_channels=self.feature_channels[0], out_channels=out_channels)

        self.level1 = DecoderBlock(
            in_channels=self.feature_channels[1],
            shared_channels=self.feature_channels[0],
            skip_channels=self.feature_channels[0],
            out_channels=self.feature_channels[0]
        )

        self.level2 = DecoderBlock(
            in_channels=self.feature_channels[2],
            shared_channels=self.feature_channels[1],
            skip_channels=self.feature_channels[1],
            out_channels=self.feature_channels[1]
        )

        self.level3 = DecoderBlock(
            in_channels=self.feature_channels[3],
            shared_channels=self.feature_channels[2],
            skip_channels=self.feature_channels[2],
            out_channels=self.feature_channels[2]
        )

        self.level4 = DecoderBlock(
            in_channels=self.feature_channels[4],
            shared_channels=self.feature_channels[3],
            skip_channels=self.feature_channels[3],
            out_channels=self.feature_channels[3]
        )

    def forward(self, ftr_stk: List[Tensor]) -> Tensor:
        out = self.level4(ftr_stk[7], ftr_stk[6], ftr_stk[5])
        out = self.level3(out, ftr_stk[4], ftr_stk[3])
        out = self.level2(out, ftr_stk[2], ftr_stk[1])
        out = self.level1(out, ftr_stk[0], ftr_stk[0])
        out = self.output_proj(out)

        return out
