import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import FlowPredictor
from src.feature_decoder import FeatureDecoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels: int = 1, branch_channels: int = 128):
        super().__init__()
        self.branch_k1 = ConvBNAct(in_channels, branch_channels, kernel_size=1)
        self.branch_k3 = ConvBNAct(in_channels, branch_channels, kernel_size=3)
        self.branch_k5 = ConvBNAct(in_channels, branch_channels, kernel_size=5)

    def forward(self, x: Tensor) -> Tensor:
        branches = [self.branch_k1(x), self.branch_k3(x), self.branch_k5(x)]
        return torch.cat(branches, dim=1)


class OutputProjection(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.intensity = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.softplus(self.intensity(features))


class Interpolator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_features = 1
        self.branch_channels = 128
        self.encoder_channels = [256, 128, 64, 32, 16]

        self.in_ftrs = FeatureExtractionBlock(self.input_features, self.branch_channels)
        self.feature_encoder = FeatureEncoder(
            self.cfg, in_channels=self.branch_channels * 3, out_channels=self.encoder_channels)
        self.flow_predictor = FlowPredictor(
            self.cfg, feature_channels=self.encoder_channels, max_disp=4)
        self.feature_decoder = FeatureDecoder(
            self.cfg, feature_channels=self.encoder_channels, out_channels=256)
        self.output_projection = OutputProjection(in_channels=256)

    def forward(self, x0: Tensor, x2: Tensor, *_, **__) -> dict[str, Tensor]:
        x0_ftr = self.in_ftrs(x0)
        x2_ftr = self.in_ftrs(x2)

        ftrs0 = self.feature_encoder(x0_ftr)
        ftrs2 = self.feature_encoder(x2_ftr)
        interpolations, warped0, warped2, _ = self.flow_predictor(ftrs0, ftrs2, x0, x2)
        decoded = self.feature_decoder(interpolations, warped0, warped2, ftrs0, ftrs2)
        pred = self.output_projection(decoded)

        return {
            "pred": pred
        }
