import os
import sys
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FlowEstimationBlock(nn.Module):
    def __init__(self, feature_channels: int, max_disp: int = 4):
        super().__init__()
        self.max_disp = max_disp
        self.search_range = 2 * max_disp + 1
        cost_channels = self.search_range ** 2

        self.flow_estimator = nn.Sequential(
            nn.Conv2d(cost_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, max(16, feature_channels // 2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(16, feature_channels // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(16, feature_channels // 2), 2, kernel_size=1),
        )
        self.blend_mask = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + 2, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def cost_volume(self, ftr0: Tensor, ftr2: Tensor) -> Tensor:
        b, c, h, w = ftr0.shape
        ftr0 = F.normalize(ftr0, dim=1)
        ftr2 = F.normalize(ftr2, dim=1)
        padded = F.pad(ftr2, [self.max_disp] * 4, mode="reflect")
        patches = F.unfold(padded, kernel_size=self.search_range)
        patches = patches.view(b, c, self.search_range ** 2, h * w)
        ftr0_flat = ftr0.view(b, c, h * w).permute(0, 2, 1)
        cost = torch.einsum("b n c, b c k n -> b k n", ftr0_flat, patches)
        return cost.view(b, self.search_range ** 2, h, w)

    @staticmethod
    def flow_to_grid(flow: Tensor, align_corners: bool = True) -> Tensor:
        b, _, h, w = flow.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=flow.device),
            torch.linspace(-1.0, 1.0, w, device=flow.device),
            indexing="ij",
        )
        base = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        denom_w = w - 1 if align_corners else w
        denom_h = h - 1 if align_corners else h
        flow_x = flow[:, 0] * (2.0 / max(denom_w, 1))
        flow_y = flow[:, 1] * (2.0 / max(denom_h, 1))
        return base + torch.stack([flow_x, flow_y], dim=-1)

    def forward(self, ftr0: Tensor, ftr2: Tensor, base_flow: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flow = self.flow_estimator(self.cost_volume(ftr0, ftr2))
        if base_flow is not None:
            flow = flow + base_flow
        grid0 = self.flow_to_grid(0.5 * flow)
        grid2 = self.flow_to_grid(-0.5 * flow)
        warped0 = F.grid_sample(ftr0, grid0, mode="bilinear", padding_mode="border", align_corners=True)
        warped2 = F.grid_sample(ftr2, grid2, mode="bilinear", padding_mode="border", align_corners=True)
        alpha = self.blend_mask(torch.cat([warped0, warped2, flow], dim=1))
        midpoint = alpha * warped0 + (1.0 - alpha) * warped2
        return midpoint, warped0, warped2, flow


class FlowPredictor(nn.Module):
    def __init__(self, cfg, feature_channels: List[int] = None, max_disp: int = 4):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = feature_channels or [256, 128, 64, 32]
        self.flow_heads = nn.ModuleList([
            FlowEstimationBlock(channels, max_disp=max(1, max_disp - idx))
            for idx, channels in enumerate(self.feature_channels)
        ])

    @staticmethod
    def _upsample_flow(flow: Tensor, size: tuple[int, int]) -> Tensor:
        _, _, h_old, w_old = flow.shape
        h_new, w_new = size
        flow = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
        flow[:, 0] *= w_new / max(w_old, 1)
        flow[:, 1] *= h_new / max(h_old, 1)
        return flow

    def forward(self, ftrs0: List[Tensor], ftrs2: List[Tensor], raw_x0: Tensor = None, raw_x2: Tensor = None):
        num_levels = len(self.flow_heads)
        interpolations = [None] * num_levels
        warps0 = [None] * num_levels
        warps2 = [None] * num_levels
        flows = [None] * num_levels
        coarse_flow = None

        for idx in reversed(range(num_levels)):
            ftr0, ftr2 = ftrs0[idx], ftrs2[idx]
            base_flow = None
            if coarse_flow is not None:
                base_flow = self._upsample_flow(coarse_flow, ftr0.shape[-2:])
            interp, warp0, warp2, flow = self.flow_heads[idx](ftr0, ftr2, base_flow)
            interpolations[idx] = interp
            warps0[idx] = warp0
            warps2[idx] = warp2
            flows[idx] = flow
            coarse_flow = flow
        return interpolations, warps0, warps2, flows
