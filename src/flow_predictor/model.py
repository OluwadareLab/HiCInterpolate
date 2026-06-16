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
            nn.Conv2d(cost_channels, feature_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 2, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        in_channels = feature_channels*2
        out_channels = feature_channels
        self.blend_mask = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def _cost_volume(self, x0: Tensor, x2: Tensor) -> Tensor:
        b, c, h, w = x0.shape
        x0 = F.normalize(x0, dim=1)
        x2 = F.normalize(x2, dim=1)
        # Use zero-padding instead of reflect to avoid artificial boundary contacts
        # for genomic Hi-C data where boundaries have specific sparsity structure
        padded = F.pad(x2, [self.max_disp] * 4, mode="constant", value=0.0)
        patches = F.unfold(padded, kernel_size=self.search_range)
        patches = patches.view(b, c, self.search_range ** 2, h * w)
        x0_flat = x0.view(b, c, h * w).permute(0, 2, 1)
        cost = torch.einsum("b n c, b c k n -> b k n", x0_flat, patches)
        return cost.view(b, self.search_range ** 2, h, w)

    def _flow_to_grid(self, flow: Tensor, align_corners: bool = True) -> Tensor:
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

    def forward(self, x0: Tensor, x2: Tensor, coarse_flow: Tensor = None) -> tuple[Tensor, Tensor]:
        flow = self.flow_estimator(self._cost_volume(x0, x2))
        if coarse_flow is not None:
            flow = flow + coarse_flow
        grid0 = self._flow_to_grid(0.5 * flow)
        grid2 = self._flow_to_grid(-0.5 * flow)
        warped0 = F.grid_sample(
            x0, grid0, mode="bilinear", padding_mode="border", align_corners=True)
        warped2 = F.grid_sample(
            x2, grid2, mode="bilinear", padding_mode="border", align_corners=True)
        blended_features = torch.cat([warped0, warped2], dim=1)
        alpha = self.blend_mask(blended_features)
        interpolated = alpha * warped0 + (1.0 - alpha) * warped2
        return interpolated, flow


class FlowPredictor(nn.Module):
    def __init__(self, feature_channels: List[int] = None, max_disp: int = 4):
        super().__init__()
        self.feature_channels = feature_channels or [16, 32, 64, 128]
        # Scale max_disp adaptively: at coarser levels (coarse indices), use smaller
        # displacements; at finer levels (fine indices), use larger to cover larger
        # absolute distances. Level 0 is finest, level -1 is coarsest.
        num_levels = len(self.feature_channels)
        self.flow_heads = nn.ModuleList([
            FlowEstimationBlock(
                channels,
                max_disp=max(2, max_disp * (2 ** (num_levels - 1 - idx)))
            )
            for idx, channels in enumerate(self.feature_channels)
        ])

    def _upsample_flow(self, flow: Tensor, size: tuple[int, int]) -> Tensor:
        _, _, h_old, w_old = flow.shape
        h_new, w_new = size
        flow = F.interpolate(
            flow, size=size, mode="bilinear", align_corners=True)
        flow[:, 0] *= w_new / max(w_old, 1)
        flow[:, 1] *= h_new / max(h_old, 1)
        return flow

    def forward(self, x0s: List[Tensor], x2s: List[Tensor]):
        num_levels = len(self.flow_heads)
        interpolations = [None] * num_levels
        coarse_flow = None
        for idx in reversed(range(num_levels)):
            x0, x2 = x0s[idx], x2s[idx]
            interpolated, flow = self.flow_heads[idx](
                x0, x2, coarse_flow)
            if idx > 0:
                coarse_flow = self._upsample_flow(flow, x0s[idx-1].shape[-2:])
            interpolations[idx] = interpolated

        return interpolations
