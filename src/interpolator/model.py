import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import FlowPredictor, FlowEstimationBlock
from src.feature_decoder import FeatureDecoder
from src.noise_block import NoiseBlock
import numpy as np
import matplotlib.pyplot as plt
import src.misc.utils as utils
from src.interpolator.shape import HiCFeatureExtractorNet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FeatureBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=kernel_size,
                      padding=kernel_size // 2,  bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeatureExtraction(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        self.branch1 = FeatureBlock(
            in_channels, out_channels, kernel_size=1)
        self.branch2 = FeatureBlock(
            in_channels, out_channels, kernel_size=3)
        self.branch3 = FeatureBlock(
            in_channels, out_channels, kernel_size=5)

    def forward(self, x: Tensor) -> Tensor:
        branches = [self.branch1(x), self.branch2(x), self.branch3(x)]
        return torch.cat(branches, dim=1)


class FeatureProjection(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3,
                      padding=3//2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=3//2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=1//2, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        projection = self.projection(x)
        return projection


class OutputProjection(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, padding=1//2, bias=False)

    def forward(self, x: Tensor, x_fine, enforce_symmetry: bool = True) -> tuple[Tensor, Tensor]:
        features = torch.cat([x, x_fine], dim=1)
        projection = self.projection(features)
        logits, mask = projection.chunk(2, dim=1)
        pred = torch.sigmoid(logits)
        pred_mask = torch.sigmoid(mask)

        if enforce_symmetry:
            pred = 0.5 * (pred + pred.transpose(-2, -1))

        return pred, pred_mask


class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_features = 1
        self.features_channels = 16
        self.encoder_channels = [16, 32, 64, 128]
        self.flow_channels = [16, 32, 64, 128]

        # self.feature_extraction = FeatureExtraction(
        #     self.input_features, self.features_channels)
        self.feature_encoder = FeatureEncoder(
            in_channels=1, out_channels=self.encoder_channels)
        self.flow_predictor = FlowPredictor(
            feature_channels=self.flow_channels, max_disp=4)
        self.feature_decoder = FeatureDecoder(
            feature_channels=self.flow_channels)
        self.feature_projection = FeatureProjection(in_channels=16)

        self.fine_interpolate = FlowEstimationBlock(
            feature_channels=1, max_disp=4)
        self.output_projection = OutputProjection(in_channels=2)

    def forward(self, x0: Tensor, x2: Tensor, enforce_input_range: bool = True, enforce_symmetry: bool = True):
        # Validate and clamp inputs to [0, 1] range for model stability
        if enforce_input_range:
            x0_min, x0_max = x0.min().item(), x0.max().item()
            x2_min, x2_max = x2.min().item(), x2.max().item()

            if x0_min < -0.01 or x0_max > 1.01 or x2_min < -0.01 or x2_max > 1.01:
                import warnings
                warnings.warn(
                    f"Input range mismatch: x0=[{x0_min:.4f}, {x0_max:.4f}], "
                    f"x2=[{x2_min:.4f}, {x2_max:.4f}]. "
                    f"Clamping to [0, 1]. Set enforce_input_range=False to skip.",
                    UserWarning
                )
                x0 = x0.clamp(0.0, 1.0)
                x2 = x2.clamp(0.0, 1.0)

        # ext0 = self.feature_extraction(x0)
        # ext2 = self.feature_extraction(x2)

        enc0 = self.feature_encoder(x0)
        enc2 = self.feature_encoder(x2)

        interpolations = self.flow_predictor(enc0, enc2)
        decoded = self.feature_decoder(interpolations)
        projection = self.feature_projection(decoded)

        # dots0, h_edges0, v_edges0 = self.shape_extractor(x0)
        # dots2, h_edges2, v_edges2 = self.shape_extractor(x2)
        # dots_interp, _ = self.interpolate(dots0, dots2)
        # h_edges_interp, _ = self.interpolate(h_edges0, h_edges2)
        # v_edges_interp, _ = self.interpolate(v_edges0, v_edges2)

        # pred = self.output_projection(
        #     projection, dots_interp, h_edges_interp, v_edges_interp)

        fine_interpolate, _ = self.fine_interpolate(x0, x2)
        pred, mask = self.output_projection(
            projection, fine_interpolate, enforce_symmetry=enforce_symmetry)

        # noise_pred = self.noise_block(x0, x2, t)
        # final_pred = pred + noise_pred

        return pred, mask
