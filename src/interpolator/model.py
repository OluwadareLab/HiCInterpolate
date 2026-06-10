import os
import sys
import torch
import torch.nn as nn
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import FlowPredictor
from src.feature_decoder import FeatureDecoder
from torch import Tensor
import torch.nn.functional as F

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

        # [SPARSITY MASK] Per-branch single-channel learnable sparsity gates
        self.mask_pixel = nn.Conv2d(feature_channels, 1, kernel_size=1)   # [SPARSITY MASK]
        self.mask_medium = nn.Conv2d(feature_channels, 1, kernel_size=1)  # [SPARSITY MASK]
        self.mask_macro = nn.Conv2d(feature_channels, 1, kernel_size=1)   # [SPARSITY MASK]

    def forward(self, x):
        feat_pixel = self.branch_pixel(x)
        feat_medium = self.branch_medium(x)
        feat_macro = self.branch_macro(x)

        # [SPARSITY MASK] Gate each scale by its own learned per-pixel presence probability
        feat_pixel = feat_pixel * torch.sigmoid(self.mask_pixel(feat_pixel))      # [SPARSITY MASK]
        feat_medium = feat_medium * torch.sigmoid(self.mask_medium(feat_medium))  # [SPARSITY MASK]
        feat_macro = feat_macro * torch.sigmoid(self.mask_macro(feat_macro))      # [SPARSITY MASK]

        out = torch.cat([feat_pixel, feat_medium, feat_macro], dim=1)
        return out


class GenomicSharpeningHead(nn.Module):
    def __init__(self):
        super().__init__()
        # 2.D Laplacian Kernel to highlight structural insulation edges
        kernel = torch.tensor([[0, -1,  0],
                               [-1,  10, -1],
                               [0, -1,  0]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))

    def forward(self, x):
        # x shape: [B, 1, 64, 64]
        # Pad edges symmetrically to preserve matrix boundary structures
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return torch.clamp(F.conv2d(x_padded, self.kernel), 0, 1)


class GenomicAffineScalingHead(nn.Module):
    def __init__(self):
        super(GenomicAffineScalingHead, self).__init__()
        # Learnable scale (gamma) initialized to 1.0, and bias (beta) initialized to 0.0
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        # Dynamically scales absolute brightness and shifts luminance
        return (x * self.gamma) + self.beta


class DenseGenomicRefinementBlock(nn.Module):
    def __init__(self, in_ch=16, out_ch=16):
        super(DenseGenomicRefinementBlock, self).__init__()
        # Parallel Multi-Scale Dilated Receptive Fields
        self.branch1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=3, dilation=3)

        self.fusion = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        # Concatenate and compress features
        fused = torch.cat([b1, b2, b3], dim=1)
        return self.leaky_relu(self.fusion(fused))


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
        self.refinement = DenseGenomicRefinementBlock(
            self.refinement_channels, 16)
        # self.refinement = nn.Sequential(
        #     nn.Conv2d(self.refinement_channels, 16,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

        # self.sharpening_head = GenomicSharpeningHead()
        self.scaling_head = GenomicAffineScalingHead()

        self.projection = nn.Conv2d(16, 1, kernel_size=1)
        self.regression_head = nn.Conv2d(16, 1, kernel_size=1)
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)



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



        # residual = self.projection(residual)
        # residual = self.sharpening_head(residual)
        residual = self.scaling_head(residual)
        # pred = residual + interpolatios[0]

        res_correction = self.regression_head(residual)
        # raw_intensity = res_correction + interpolatios[0]
        raw_intensity = res_correction
        
        # Apply ReLU or Softplus to ensure intensity is physically non-negative
        predicted_intensity = F.softplus(raw_intensity)
        
        # 2. Calculate Sparsity Mask (Probability Space)
        # Sigmoid squashes output to [0, 1] range
        mask_logits = self.mask_head(residual)
        predicted_mask_prob = torch.sigmoid(mask_logits)
        
        # 3. Final Gated Output (The "Filtered" Result)
        # During training, we use the probability. 
        # During inference, we can use a hard threshold (e.g., > 0.5)
        final_output = predicted_intensity * predicted_mask_prob
        
        return {
            "final": final_output,          # Use this for HiCRep/SSIM/etc.
            "mask_prob": predicted_mask_prob, # Use this for metrics/inference
            "intensity": predicted_intensity, # Intermediate regression
            "mask_logits": mask_logits        # Use this for stable BCE-with-logits
        }


        return residual
