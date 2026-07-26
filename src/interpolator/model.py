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
    def __init__(self, in_channels: int = 4, hidden_channels: int = 32):
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

    def forward(self, x: Tensor, x0: Tensor, x2: Tensor,
                enforce_symmetry: bool = True) -> tuple[Tensor, Tensor]:
        support_prior = torch.maximum(x0, x2).clamp(0.0, 1.0)
        diag_dist = diagonal_distance_channel(x0)
        features = torch.cat([
            x,
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
        pred_mask = torch.maximum(
            support_prior, learned_support * local_support)
        intensity = torch.sigmoid(self.intensity_head(hidden))
        residual = torch.tanh(self.residual_head(hidden))
        pred = pred_mask * (0.5 * intensity + 0.25 * residual)
        pred = pred.clamp(0.0, 1.0)

        if enforce_symmetry:
            pred = 0.5 * (pred + pred.transpose(-2, -1))
            pred_mask = 0.5 * (pred_mask + pred_mask.transpose(-2, -1))

        return pred, pred_mask


class MultiScaleBlockAttention(nn.Module):
    def __init__(self, channels, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.coarse_attn = nn.Conv2d(channels, 1, kernel_size=block_size,
                                     stride=block_size, padding=0)
        self.fine_attn = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Coarse block-level attention (TAD scale)
        coarse = torch.sigmoid(self.coarse_attn(x))
        coarse_up = F.interpolate(coarse, size=(H, W), mode='nearest')

        # Fine pixel-level attention
        fine = torch.sigmoid(self.fine_attn(x))

        # Combine: coarse gates which TAD blocks are active
        combined = coarse_up * fine
        return x * combined


class SparseAttentionMask(nn.Module):
    def __init__(self, channels, threshold=1e-4):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, 1)
        self.threshold = threshold

    def forward(self, x, x0, x2):
        # Nonzero mask from input boundary frames
        nz_mask = ((x0 > self.threshold) | (x2 > self.threshold)).float()

        # Soft learned attention
        soft_attn = torch.sigmoid(self.attn(x))

        # Hard-gate: zero out attention in regions both inputs agree are empty
        masked_attn = soft_attn * nz_mask

        return x * masked_attn


class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_features = 1
        self.base_channels = 32
        self.depth = 4
        self.feature_encoder = FeatureEncoder(
            input_channels=self.input_features, base_channels=self.base_channels, depth=self.depth)
        self.flow_predictor = FlowPredictor(
            base_channels=self.base_channels, depth=self.depth+1, max_disp=4)
        self.feature_decoder = FeatureDecoder(
            base_channels=self.base_channels, depth=self.depth)

        self.output_projection = OutputProjection(
            in_channels=6, hidden_channels=32)
        self.multiscale_attention = MultiScaleBlockAttention(
            channels=self.base_channels)
        self.sparse_attention = SparseAttentionMask(
            channels=self.base_channels)
        self.residual_head = nn.Conv2d(self.base_channels, 1,
                                       kernel_size=1, padding=0, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, x0: Tensor, x2: Tensor):
        enc0 = self.feature_encoder(x0)
        enc2 = self.feature_encoder(x2)

        interpolations = self.flow_predictor(enc0, enc2)
        decoded = self.feature_decoder(interpolations)

        # multiscale_attended = self.multiscale_attention(decoded)
        # sparse_attended = self.sparse_attention(multiscale_attended, x0, x2)
        projection = self.residual_head(decoded)

        # squares0, dots0, h_edges0, v_edges0 = utils.image_segmentation_batch(
        #     x0)
        # squares2, dots2, h_edges2, v_edges2 = utils.image_segmentation_batch(
        #     x2)
        # pred, mask = self.output_projection(
        #     projection, dots0, dots2)
        projection[projection < 0] = 0
        return projection
