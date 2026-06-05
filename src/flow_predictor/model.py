import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FlowEstimation(nn.Module):
    def __init__(self, feature_channels=128, max_disp=4):
        super().__init__()
        self.max_disp = max_disp
        self.search_range = 2 * max_disp + 1
        self.out_channels = self.search_range**2
        self.ftr_ext = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
        )
        corr_channels = (2 * max_disp + 1) ** 2
        self.process_head = nn.Sequential(
            nn.Conv2d(corr_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
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

        # if ftr2.shape[2] > max_disp and ftr2.shape[3] > max_disp:
        #     padded_ftr2 = F.pad(
        #         ftr2,
        #         [max_disp, max_disp, max_disp, max_disp],
        #         mode="reflect"
        #     )
        # else:
        #     padded_ftr2 = F.pad(
        #         ftr2,
        #         [max_disp, max_disp, max_disp, max_disp],
        #         mode="replicate"
        #     )

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
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )

        return warped_output


class ForwardFlowPredictor(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], max_disp=5):
        super().__init__()
        self.cfg = cfg
        self.proj_flow = FlowEstimation(
            feature_channels=feature_channels[0], max_disp=max_disp)
        self.unique_flow1 = FlowEstimation(
            feature_channels=feature_channels[1], max_disp=max_disp)
        self.shared_flow1 = FlowEstimation(
            feature_channels=feature_channels[1], max_disp=max_disp)
        self.unique_flow2 = FlowEstimation(
            feature_channels=feature_channels[2], max_disp=max_disp-1)
        self.shared_flow2 = FlowEstimation(
            feature_channels=feature_channels[2], max_disp=max_disp-1)
        self.unique_flow3 = FlowEstimation(
            feature_channels=feature_channels[3], max_disp=max_disp-2)
        self.shared_flow3 = FlowEstimation(
            feature_channels=feature_channels[3], max_disp=max_disp-2)
        self.unique_flow4 = FlowEstimation(
            feature_channels=feature_channels[4], max_disp=max_disp-2)
        self.shared_flow4 = FlowEstimation(
            feature_channels=feature_channels[4], max_disp=max_disp-2)
        self.flow_heads = [self.proj_flow, self.unique_flow1, self.shared_flow1,
                           self.unique_flow2, self.shared_flow2, self.unique_flow3, self.shared_flow3,
                           self.unique_flow4, self.shared_flow4]

    def forward(self, ftr0_stk: list[torch.Tensor], ftr2_stk: list[torch.Tensor], time: torch.Tensor):
        forward_flows = []
        for ftr0, ftr2, flow_head in zip(ftr0_stk, ftr2_stk, self.flow_heads):
            warped_output = flow_head(ftr0, ftr2, time)
            forward_flows.append(warped_output)

        return forward_flows


class BackwardFlowPredictor(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256, 512], max_disp=5):
        super().__init__()
        self.cfg = cfg
        self.proj_flow = FlowEstimation(
            feature_channels=feature_channels[0], max_disp=max_disp)
        self.unique_flow1 = FlowEstimation(
            feature_channels=feature_channels[1], max_disp=max_disp)
        self.shared_flow1 = FlowEstimation(
            feature_channels=feature_channels[1], max_disp=max_disp)
        self.unique_flow2 = FlowEstimation(
            feature_channels=feature_channels[2], max_disp=max_disp-1)
        self.shared_flow2 = FlowEstimation(
            feature_channels=feature_channels[2], max_disp=max_disp-1)
        self.unique_flow3 = FlowEstimation(
            feature_channels=feature_channels[3], max_disp=max_disp-2)
        self.shared_flow3 = FlowEstimation(
            feature_channels=feature_channels[3], max_disp=max_disp-2)
        self.unique_flow4 = FlowEstimation(
            feature_channels=feature_channels[4], max_disp=max_disp-2)
        self.shared_flow4 = FlowEstimation(
            feature_channels=feature_channels[4], max_disp=max_disp-2)
        self.flow_heads = [self.proj_flow, self.unique_flow1, self.shared_flow1,
                           self.unique_flow2, self.shared_flow2, self.unique_flow3, self.shared_flow3,
                           self.unique_flow4, self.shared_flow4]

    def forward(self, ftr2_stk: list[torch.Tensor], ftr0_stk: list[torch.Tensor], time: torch.Tensor):
        backward_flows = []
        for ftr0, ftr2, flow_head in zip(ftr0_stk, ftr2_stk, self.flow_heads):
            warped_output = flow_head(ftr0, ftr2, time)
            backward_flows.append(warped_output)

        return backward_flows
