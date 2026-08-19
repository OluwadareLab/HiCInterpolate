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


class BlockAttnetion(nn.Module):
    def __init__(self, channels, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.coarse_attn = nn.Conv2d(channels, 1, kernel_size=block_size,
                                     stride=block_size, padding=0)
        self.fine_attn = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        coarse = torch.sigmoid(self.coarse_attn(x))
        coarse_up = F.interpolate(coarse, size=(H, W), mode='nearest')

        fine = torch.sigmoid(self.fine_attn(x))
        combined = coarse_up * fine
        return x * combined


class ProjectionGate(nn.Module):
    def __init__(self, channels, threshold=1e-4):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, 1)
        self.threshold = threshold

    def forward(self, x, x0, x2):
        nz_mask = ((x0 > self.threshold) | (x2 > self.threshold)).float()
        soft_attn = torch.sigmoid(self.attn(x))
        masked_attn = soft_attn * nz_mask

        return x * masked_attn


class Interpolator(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.input_features = 1
        self.base_channels = 32
        self.depth = 4
        self.feature_encoder = FeatureEncoder(
            input_channels=self.input_features, base_channels=self.base_channels, depth=self.depth, dropout=dropout)
        self.flow_predictor = FlowPredictor(
            base_channels=self.base_channels, depth=self.depth+1, max_disp=4)
        self.feature_decoder = FeatureDecoder(
            base_channels=self.base_channels, depth=self.depth, dropout=dropout)

        self.multiscale_attention = BlockAttnetion(
            channels=self.base_channels)
        self.sparse_attention = ProjectionGate(
            channels=self.base_channels)
        self.residual_head = nn.Conv2d(self.base_channels, 1,
                                       kernel_size=1, padding=0, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, x0: Tensor, x2: Tensor):
        enc0 = self.feature_encoder(x0)
        enc2 = self.feature_encoder(x2)

        interpolations = self.flow_predictor(enc0, enc2)
        decoded = self.feature_decoder(interpolations)

        multiscale_attended = self.multiscale_attention(decoded)
        sparse_attended = self.sparse_attention(multiscale_attended, x0, x2)
        projection = self.residual_head(sparse_attended)
        projection[projection < 0] = 0
        return projection
