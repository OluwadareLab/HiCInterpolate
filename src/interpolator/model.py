import os
import sys
import torch
import torch.nn as nn
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import ForwardFlow, BackwardFlow
from src.feature_decoder import FeatureDecoder
from torch import Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


class Interpolator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_channels = 1
        self.feature_channels = list([32, 64, 128, 256, 512])
        self.out_channels = 1
        self.feature_encoder = FeatureEncoder(
            self.cfg, in_channels=self.in_channels, out_channels=self.feature_channels)
        self.forward_flow = ForwardFlow(
            self.cfg, feature_channels=self.feature_channels)
        self.backward_flow = BackwardFlow(
            self.cfg, feature_channels=self.feature_channels)
        self.feature_decoder = FeatureDecoder(self.cfg)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.feature_channels[0],
                      kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(self.feature_channels[0]),
            nn.Conv2d(self.feature_channels[0], self.feature_channels[0],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feature_channels[0])
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_channels[0], out_channels=self.feature_channels[0]//2,
                      kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(self.feature_channels[0]//2),
            nn.Conv2d(self.feature_channels[0]//2, self.out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    @staticmethod
    def concatenate_flow_ftr(ftr_0: list[Tensor], ftr_2: list[Tensor]) -> list[Tensor]:
        mid_ftr = []
        for feature1, feature2 in zip(ftr_0, ftr_2):
            mid_ftr.append(0.5 * (feature1 + feature2))
        return mid_ftr

    def forward(self, x0: Tensor, x2: Tensor, time: Tensor) -> Tensor:
        # Feature Encoder
        x0 = self.in_proj(x0)
        x2 = self.in_proj(x2)
        ftrs0 = self.feature_encoder(x0)
        ftrs2 = self.feature_encoder(x2)

        # Flow Predictor
        forward_mid_ftrs = self.forward_flow(
            ftrs0, ftrs2, time[:, 0])
        backward_mid_ftrs = self.backward_flow(
            ftrs2, ftrs0, time[:, 0])

        # Feature Alignment
        mid_ftrs = self.concatenate_flow_ftr(
            forward_mid_ftrs, backward_mid_ftrs)

        # Feature Decoder
        residual = self.feature_decoder(mid_ftrs)
        pred = self.out_proj(residual + mid_ftrs[0])

        return pred
