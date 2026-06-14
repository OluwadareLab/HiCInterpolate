import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

class MorphologicalSegregation(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def erosion(self, x: Tensor) -> Tensor:
        return -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def dilation(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def opening(self, x: Tensor) -> Tensor:
        return self.dilation(self.erosion(x))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        i_struct = self.opening(x)
        i_noise = torch.abs(x - i_struct)
        return i_struct, i_noise

class PWCFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        return [c1, c2, c3]

class CostVolumeLayer(nn.Module):
    def __init__(self, max_disp: int = 4):
        super().__init__()
        self.max_disp = max_disp
        self.search_range = 2 * max_disp + 1

    def forward(self, ftr0: Tensor, ftr1: Tensor) -> Tensor:
        b, c, h, w = ftr0.shape
        padded = F.pad(ftr1, [self.max_disp] * 4, mode="reflect")
        patches = F.unfold(padded, kernel_size=self.search_range)
        patches = patches.view(b, c, self.search_range ** 2, h * w)
        ftr0_flat = ftr0.view(b, c, h * w).permute(0, 2, 1)
        cost = torch.einsum("b n c, b c k n -> b k n", ftr0_flat, patches)
        return cost.view(b, self.search_range ** 2, h, w)

class BottleneckUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.enc1(x)
        x = self.enc2(x)
        latent = self.bottleneck(x)
        x = self.dec2(latent)
        x = self.dec1(x)
        return x, latent

class NoiseBlock(nn.Module):
    def __init__(self, kernel_size: int = 3, max_disp: int = 4):
        super().__init__()
        self.segregation = MorphologicalSegregation(kernel_size)
        self.feature_extractor = PWCFeatureExtractor()
        self.cost_volume = CostVolumeLayer(max_disp)
        
        cost_channels = (2 * max_disp + 1) ** 2
        self.flow_predictor = BottleneckUNet(cost_channels, 4)
        
        # Decoder inputs: warped structures (2) + latents (interpolated to full size)
        # Latents from bottleneck are 128 channels at 1/4 resolution (if input was 1/2 from PWC)
        # Wait, PWC conv1 is stride 2. Cost volume is at 1/2 resolution.
        # Bottleneck is at 1/8 resolution.
        self.struct_decoder = nn.Sequential(
            nn.Conv2d(2 + 128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, kernel_size=1) # [struct, mask]
        )

    def warp(self, x: Tensor, flow: Tensor) -> Tensor:
        b, _, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        flow_x = flow[:, 0] * (2.0 / max(w - 1, 1))
        flow_y = flow[:, 1] * (2.0 / max(h - 1, 1))
        grid = grid + torch.stack([flow_x, flow_y], dim=-1)
        
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, i0: Tensor, i1: Tensor, t: float = 0.5) -> Tensor:
        # 1. Morphological Segregation
        i0_struct, i0_noise = self.segregation(i0)
        i1_struct, i1_noise = self.segregation(i1)
        
        # 2. PWC Feature Extraction
        ftrs0 = self.feature_extractor(i0_struct)
        ftrs1 = self.feature_extractor(i1_struct)
        
        # 3. Cost Volume (Scale 1)
        cost = self.cost_volume(ftrs0[0], ftrs1[0])
        
        # 4. Flow Prediction & Latents
        flows, latent = self.flow_predictor(cost)
        full_flows = F.interpolate(flows, size=i0.shape[-2:], mode="bilinear", align_corners=True)
        f_t0, f_t1 = full_flows.chunk(2, dim=1)
        
        # 5. Temporal Interpolation & Warping
        i0_struct_w = self.warp(i0_struct, f_t0)
        i1_struct_w = self.warp(i1_struct, f_t1)
        i0_noise_w = self.warp(i0_noise, f_t0)
        i1_noise_w = self.warp(i1_noise, f_t1)
        
        # 6. Decoder & Mask Blending
        latent_up = F.interpolate(latent, size=i0.shape[-2:], mode="bilinear", align_corners=True)
        struct_mask = self.struct_decoder(torch.cat([i0_struct_w, i1_struct_w, latent_up], dim=1))
        i_t_struct, m_t = struct_mask.chunk(2, dim=1)
        i_t_struct = torch.sigmoid(i_t_struct)
        m_t = torch.sigmoid(m_t)
        
        # Unified noise field
        i_t_noise = (1 - t) * i0_noise_w + t * i1_noise_w
        
        # Final Output Synthesis
        i_t = m_t * i_t_struct + (1 - m_t) * i_t_noise
        return i_t
