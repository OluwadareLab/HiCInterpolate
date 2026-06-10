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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device))
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )
        args = t.float().view(-1, 1) * freq.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SparseDiffusionRefinementBlock(nn.Module):
    def __init__(self, feature_channels=16, hidden_channels=32, time_channels=32):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_channels),
            nn.Linear(time_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.input_proj = nn.Conv2d(feature_channels + 2, hidden_channels, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.noise_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, features, noisy_x, base_x, timestep):
        h = self.input_proj(torch.cat([features, noisy_x, base_x], dim=1))
        t_emb = self.time_embedding(timestep).view(timestep.size(0), -1, 1, 1)
        h = h + t_emb
        h = h + self.refine(h)
        return self.noise_head(h)


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

        diffusion_cfg = getattr(self.cfg.model, "diffusion", None)
        self.diffusion_enabled = bool(getattr(diffusion_cfg, "enabled", True))
        self.diffusion_timesteps = int(getattr(diffusion_cfg, "timesteps", 64))
        beta_start = float(getattr(diffusion_cfg, "beta_start", 1e-4))
        beta_end = float(getattr(diffusion_cfg, "beta_end", 2e-2))
        hidden_channels = int(getattr(diffusion_cfg, "hidden_channels", 32))
        self.inference_timestep = int(getattr(diffusion_cfg, "inference_timestep", 4))
        self.preserve_input_support = bool(getattr(diffusion_cfg, "preserve_input_support", True))
        betas = torch.linspace(beta_start, beta_end, self.diffusion_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.diffusion_refiner = SparseDiffusionRefinementBlock(
            feature_channels=16, hidden_channels=hidden_channels)

    @staticmethod
    def concatenate_flow_ftr(ftr_0: list[Tensor], ftr_2: list[Tensor]) -> list[Tensor]:
        mid_ftr = []
        for feature1, feature2 in zip(ftr_0, ftr_2):
            mid_ftr.append(torch.cat([feature1, feature2], dim=1))
        return mid_ftr

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.diffusion_timesteps, (batch_size,), device=device)

    def _q_sample_sparse(self, clean: Tensor, support: Tensor, timesteps: Tensor):
        noise = torch.randn_like(clean) * support
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        noisy = sqrt_alpha * clean + sqrt_one_minus * noise
        noisy = noisy * support
        return noisy, noise

    def _predict_x0_from_noise(self, noisy: Tensor, noise_pred: Tensor, timesteps: Tensor) -> Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).clamp_min(1e-6)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return (noisy - sqrt_one_minus * noise_pred) / sqrt_alpha

    def _diffusion_refine(self, features: Tensor, base_intensity: Tensor, support: Tensor, timestep_value: int):
        timestep_value = max(0, min(timestep_value, self.diffusion_timesteps - 1))
        timesteps = torch.full(
            (base_intensity.size(0),), timestep_value,
            device=base_intensity.device, dtype=torch.long)
        noise_pred = self.diffusion_refiner(features, base_intensity * support, base_intensity, timesteps)
        refined = self._predict_x0_from_noise(base_intensity, noise_pred * support, timesteps)
        return refined.clamp_min(0.0) * support, noise_pred, timesteps

    def forward(self, x0: Tensor, x2: Tensor, time: Tensor, target: Tensor = None) -> Tensor:
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
        raw_intensity = res_correction
        base_intensity = F.softplus(raw_intensity)

        mask_logits = self.mask_head(residual)
        predicted_mask_prob = torch.sigmoid(mask_logits)

        support_prob = predicted_mask_prob
        if self.preserve_input_support:
            endpoint_support = ((x0 > 0) | (x2 > 0)).float()
            support_prob = torch.maximum(support_prob, endpoint_support)

        predicted_intensity = base_intensity
        diffusion_noise_pred = None
        diffusion_noise_target = None
        diffusion_mask = None
        diffusion_t = None

        if self.diffusion_enabled:
            predicted_intensity, _, _ = self._diffusion_refine(
                residual, base_intensity, support_prob, self.inference_timestep)

            if target is not None:
                diffusion_mask = (target > 0).float()
                diffusion_t = self._sample_timesteps(target.size(0), target.device)
                noisy_target, diffusion_noise_target = self._q_sample_sparse(
                    target, diffusion_mask, diffusion_t)
                diffusion_noise_pred = self.diffusion_refiner(
                    residual, noisy_target, base_intensity.detach(), diffusion_t) * diffusion_mask
                diffusion_noise_target = diffusion_noise_target * diffusion_mask

        final_output = predicted_intensity * predicted_mask_prob

        return {
            "final": final_output,
            "mask_prob": predicted_mask_prob,
            "intensity": predicted_intensity,
            "base_intensity": base_intensity,
            "mask_logits": mask_logits,
            "diffusion_noise_pred": diffusion_noise_pred,
            "diffusion_noise_target": diffusion_noise_target,
            "diffusion_mask": diffusion_mask,
            "diffusion_t": diffusion_t,
        }
