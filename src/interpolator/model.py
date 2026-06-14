import os
import sys
import time
import torch
import torch.nn as nn
from torch import Tensor
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import FlowPredictor
from src.feature_decoder import FeatureDecoder
from src.noise_block import NoiseBlock
import numpy as np
import matplotlib.pyplot as plt
import src.misc.utils as utils
from src.interpolator.shape import HiCFeatureExtractorNet
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
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        intensity_logits, mask_logits = self.intensity(
            features).chunk(2, dim=1)
        intensity = torch.sigmoid(intensity_logits)
        mask = torch.sigmoid(mask_logits)
        return intensity * mask, mask


class Interpolator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_features = 1
        self.branch_channels = 128
        self.encoder_channels = [256, 128, 64, 32, 16]

        # self.in_ftrs = FeatureExtractionBlock(
        #     self.input_features, self.branch_channels)
        # self.feature_encoder = FeatureEncoder(
        #     self.cfg, in_channels=self.branch_channels * 3, out_channels=self.encoder_channels)
        # self.flow_predictor = FlowPredictor(
        #     self.cfg, feature_channels=self.encoder_channels, max_disp=4)
        # self.feature_decoder = FeatureDecoder(
        #     self.cfg, feature_channels=self.encoder_channels, out_channels=256)
        # self.output_projection = OutputProjection(in_channels=256)
        # self.noise_block = NoiseBlock(kernel_size=3, max_disp=4)
        self.shape_extractor = HiCFeatureExtractorNet(in_channels=1, base_channels=64)

    def forward(self, x0: Tensor, x2: Tensor, *args, **kwargs) -> dict[str, Tensor]:

        # square0, dots0, h_edges0, v_edges0 = utils.image_segmentation_batch(x0)
        # square2, dots2, h_edges2, v_edges2 = utils.image_segmentation_batch(x2)

        square0_cuda, dots0_cuda, h_edges0_cuda, v_edges0_cuda = utils.image_segmentation_cuda_approx(x0)
        square2_cuda, dots2_cuda, h_edges2_cuda, v_edges2_cuda = utils.image_segmentation_cuda_approx(x2)

        pred_dots  = self.shape_extractor(dots0_cuda, dots2_cuda)

        # call noise block

        return pred_dots
        # t = 0.5
        # if len(args) > 0:
        #     t = args[0]
        # elif "time_frame" in kwargs:
        #     t = kwargs["time_frame"]

        # x0_ftr = self.in_ftrs(x0)
        # x2_ftr = self.in_ftrs(x2)

        # ftrs0 = self.feature_encoder(x0_ftr)
        # ftrs2 = self.feature_encoder(x2_ftr)
        # interpolations, warped0, warped2, _ = self.flow_predictor(ftrs0, ftrs2, x0, x2)
        # decoded = self.feature_decoder(interpolations, warped0, warped2, ftrs0, ftrs2)
        # pred, mask = self.output_projection(decoded)

        # noise_pred = self.noise_block(x0, x2, t)

        # # Combine base prediction and noise block output
        # # uniquely learn and work on top of current architecture
        # final_pred = pred + noise_pred

        # return {
        #     "pred": noise_pred,
        #     "mask": mask,
        #     "noise_pred": noise_pred,
        # }
