import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FlowEstimationBlock(nn.Module):
    def __init__(self, feature_channels=128, max_disp=4):
        super().__init__()
        self.max_disp = max_disp
        self.search_range = 2 * max_disp + 1
        self.out_channels = self.search_range**2
        self.ftr_ext = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(),
        )
        corr_channels = (2 * max_disp + 1) ** 2
        self.process_head = nn.Sequential(
            nn.Conv2d(corr_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    @staticmethod
    def cost_volume_estimator(ftr0, ftr2, max_disp, search_range, out_channels):
        B, C, H, W = ftr0.shape
        ftr0 = F.normalize(ftr0, dim=1)
        ftr2 = F.normalize(ftr2, dim=1)

        padded_ftr2 = F.pad(
            ftr2,
            [max_disp, max_disp, max_disp, max_disp],
            mode="reflect"
        )

        patches = F.unfold(
            padded_ftr2,
            kernel_size=(search_range, search_range),
            padding=0,
            stride=1,
        )
        patches = patches.view(B, C, out_channels, H * W)
        ftr0_flat = ftr0.view(B, C, H * W).permute(0, 2, 1)
        cost_volume = torch.einsum(
            "b i c, b c k i -> b k i", ftr0_flat, patches)
        return cost_volume.view(B, out_channels, H, W)

    @staticmethod
    def flow_to_warp_grid(flow, align_corners=True):
        B, _, H, W = flow.shape
        device = flow.device

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij",
        )

        identity_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        identity_grid = identity_grid.repeat(B, 1, 1, 1)
        u_normalized = flow[:, 0, :, :] * \
            (2.0 / (W - 1 if align_corners else W))
        v_normalized = flow[:, 1, :, :] * \
            (2.0 / (H - 1 if align_corners else H))
        normalized_flow = torch.stack([u_normalized, v_normalized], dim=-1)
        transformation_grid = identity_grid + normalized_flow

        return transformation_grid

    def forward(self, src_img, tgt_img, time):
        src_ftr = self.ftr_ext(src_img)
        tgt_ftr = self.ftr_ext(tgt_img)
        corr_map = self.cost_volume_estimator(
            src_ftr, tgt_ftr, self.max_disp, self.search_range, self.out_channels)
        flow_field = self.process_head(corr_map)
        timed_flow_field = flow_field * time.view(-1, 1, 1, 1)
        warp_grid = self.flow_to_warp_grid(
            timed_flow_field, align_corners=True)
        warped_output = F.grid_sample(
            src_img,
            warp_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped_output


class FlowPredictor(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], max_disp=5):
        super().__init__()
        self.cfg = cfg
        self.flow_heads = nn.ModuleList([
            FlowEstimationBlock(
                feature_channels=feature_channels[i],
                max_disp=self._max_disp_for_level(i, max_disp),
            )
            for i in range(len(feature_channels))
        ])

    @staticmethod
    def _max_disp_for_level(level_idx: int, max_disp: int) -> int:
        if level_idx < 2:
            return max_disp
        if level_idx == 2:
            return max_disp - 1
        return max_disp - 2

    def forward(self, ftr0_stk: list[torch.Tensor], ftr2_stk: list[torch.Tensor], time: torch.Tensor):
        forward_flows = []
        for ftr0, ftr2, flow_head in zip(ftr0_stk, ftr2_stk, self.flow_heads):
            warped_output = flow_head(ftr0, ftr2, time)
            forward_flows.append(warped_output)

        return forward_flows


class ForwardFlow(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], max_disp=5):
        super().__init__()
        self.flow_pred = FlowPredictor(cfg, feature_channels, max_disp)

    def forward(self, ftr0_stk: list[torch.Tensor], ftr2_stk: list[torch.Tensor], time: torch.Tensor):
        forward_flows = self.flow_pred(ftr0_stk, ftr2_stk, time)
        return forward_flows


class BackwardFlow(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], max_disp=5):
        super().__init__()
        self.flow_pred = FlowPredictor(cfg, feature_channels, max_disp)

    def forward(self, ftr0_stk: list[torch.Tensor], ftr2_stk: list[torch.Tensor], time: torch.Tensor):
        backward_flows = self.flow_pred(ftr0_stk, ftr2_stk, time)
        return backward_flows
