import os
import sys
import torch
import torch.nn as nn
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import FlowPredictor
from src.feature_decoder import FeatureDecoder
from torch import Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        self.branch_pixel = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_medium = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_macro = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels,
                      kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat_pixel = self.branch_pixel(x)
        feat_medium = self.branch_medium(x)
        feat_macro = self.branch_macro(x)
        out = torch.cat([feat_pixel, feat_medium, feat_macro], dim=1)
        return out


class Interpolator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_features = 1
        self.init_ftr_channels = 16
        self.in_ftrs = FeatureExtractionBlock(
            self.input_features, self.init_ftr_channels)

        self.enc_ftr_channels = list([32, 64, 128, 256])
        self.feature_encoder = FeatureEncoder(
            self.cfg, in_channels=3*self.init_ftr_channels, out_channels=self.enc_ftr_channels)

        self.flow_ftr_channels = list([32, 64, 128, 256])
        self.flow_predictor = FlowPredictor(
            self.cfg, feature_channels=self.flow_ftr_channels, max_disp=4)

        self.dec_ftr_channels = list([32, 64, 128, 256])
        self.output_features = 1
        self.feature_decoder = FeatureDecoder(
            self.cfg, feature_channels=self.dec_ftr_channels, out_channels=self.output_features)

        self.refinement_channels = self.dec_ftr_channels[0]
        self.refinement = nn.Sequential(
            nn.Conv2d(self.refinement_channels, 16,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.projection = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def concatenate_flow_ftr(ftr_0: list[Tensor], ftr_2: list[Tensor]) -> list[Tensor]:
        mid_ftr = []
        for feature1, feature2 in zip(ftr_0, ftr_2):
            mid_ftr.append(torch.cat([feature1, feature2], dim=1))
        return mid_ftr

    def forward(self, x0: Tensor, x2: Tensor, time: Tensor) -> Tensor:
        # feature extractor
        x0_ftr = self.in_ftrs(x0)  # channels = 16*3 = 48
        x2_ftr = self.in_ftrs(x2)  # channels = 16*3 = 48
        # feature encoder
        # ftrs0 = [x0_ftr]
        # ftrs0.extend(self.feature_encoder(x0_ftr))
        # ftrs2 = [x2_ftr]
        # ftrs2.extend(self.feature_encoder(x2_ftr))

        ftrs0 = self.feature_encoder(x0_ftr)
        ftrs2 = self.feature_encoder(x2_ftr)

        # Flow Predictor
        interpolatios, warped0, warped2 = self.flow_predictor(
            ftrs0, ftrs2, x0, x2)

        # Feature Decoder
        residual = self.feature_decoder(
            ftrs0, ftrs2, interpolatios, warped0, warped2)
        residual = self.refinement(residual)
        residual = self.projection(residual)
        # pred = residual + interpolatios[0]

        return residual
