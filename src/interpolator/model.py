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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def diagonal_distance_channel(x: Tensor) -> Tensor:
    b, _, h, w = x.shape
    rows = torch.arange(h, device=x.device, dtype=x.dtype).view(h, 1)
    cols = torch.arange(w, device=x.device, dtype=x.dtype).view(1, w)
    dist = (rows - cols).abs() / max(h - 1, 1)
    return dist.view(1, 1, h, w).expand(b, 1, h, w)


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
    def __init__(self, in_channels: int = 7, hidden_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.intensity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1, groups=hidden_channels,
                      bias=False),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: Tensor, x_fine: Tensor, x0: Tensor, x2: Tensor,
                enforce_symmetry: bool = True) -> tuple[Tensor, Tensor]:
        support_prior = torch.maximum(x0, x2).clamp(0.0, 1.0)
        diag_dist = diagonal_distance_channel(x0)
        features = torch.cat([
            x,
            x_fine,
            x0,
            x2,
            (x2 - x0).abs(),
            support_prior,
            diag_dist,
        ], dim=1)
        hidden = self.stem(features)
        learned_support = torch.sigmoid(self.support_head(hidden))
        local_support = F.max_pool2d(
            support_prior, kernel_size=3, stride=1, padding=1)
        pred_mask = torch.maximum(support_prior, learned_support * local_support)
        intensity = torch.sigmoid(self.intensity_head(hidden))
        residual = torch.tanh(self.residual_head(hidden))
        pred = pred_mask * (0.5 * x_fine + 0.5 * intensity + 0.25 * residual)
        pred = pred.clamp(0.0, 1.0)

        if enforce_symmetry:
            pred = 0.5 * (pred + pred.transpose(-2, -1))
            pred_mask = 0.5 * (pred_mask + pred_mask.transpose(-2, -1))

        return pred, pred_mask


class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_features = 1
        self.features_channels = 16
        self.encoder_channels = [16, 32, 64, 128]
        self.flow_channels = [16, 32, 64, 128]


        self.feature_encoder = FeatureEncoder(
            in_channels=1, out_channels=self.encoder_channels)
        self.flow_predictor = FlowPredictor(
            feature_channels=self.flow_channels, max_disp=4)
        self.feature_decoder = FeatureDecoder(
            feature_channels=self.flow_channels)
        self.feature_projection = FeatureProjection(in_channels=16)

        self.fine_interpolate = FlowEstimationBlock(
            feature_channels=1, max_disp=4)
        self.output_projection = OutputProjection(in_channels=7)

    def forward(self, x0: Tensor, x2: Tensor, enforce_input_range: bool = True, enforce_symmetry: bool = True):


        enc0 = self.feature_encoder(x0)
        enc2 = self.feature_encoder(x2)

        interpolations = self.flow_predictor(enc0, enc2)
        decoded = self.feature_decoder(interpolations)
        projection = self.feature_projection(decoded)

        fine_interpolate, _ = self.fine_interpolate(x0, x2)
        pred, mask = self.output_projection(
            projection, fine_interpolate, x0, x2,
            enforce_symmetry=enforce_symmetry)

        return pred, mask
