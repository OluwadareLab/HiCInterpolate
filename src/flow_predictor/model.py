import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class FlowEstimationBlock(nn.Module):
    def __init__(self, feature_channels=128, max_disp=4):
        super().__init__()
        search_range = (2 * max_disp + 1)**2
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(search_range, 64, kernel_size=3,
                      padding=1, bias=False),
            gn(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            gn(32),
            nn.LeakyReLU(0.2, inplace=True),
            # Outputs U and V flow components
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )

        self.mask_estimator = nn.Sequential(
            nn.Conv2d(feature_channels * 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
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

    def forward(self, ftr0, ftr1, x0, x1):
        cve = self.cost_volume_estimator(
            ftr0, ftr1, max_disp=4, search_range=9, out_channels=81)
        flow_0_to_1 = self.flow_estimator(cve)

        flow_f0_t05 = flow_0_to_1 * 0.5
        flow_f1_t05 = -flow_0_to_1 * 0.5

        grid_0 = self.flow_to_warp_grid(
            flow_f0_t05, align_corners=True)
        grid_1 = self.flow_to_warp_grid(
            flow_f1_t05, align_corners=True)

        warped_x0 = F.grid_sample(
            x0, grid_0, mode="bilinear", padding_mode="zeros", align_corners=True)
        warped_x1 = F.grid_sample(
            x1, grid_1, mode="bilinear", padding_mode="zeros", align_corners=True)

        mask = self.mask_estimator(torch.cat([ftr0, ftr1], dim=1))
        interpolated_ftr = mask * warped_x0 + (1 - mask) * warped_x1
        return interpolated_ftr, warped_x0, warped_x1


class FlowPredictor(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256], max_disp=4):
        super().__init__()
        self.cfg = cfg
        self.flow_heads = nn.ModuleList([
            FlowEstimationBlock(
                feature_channels=feature_channels[i],
                max_disp=max_disp,
            )
            for i in range(len(feature_channels))
        ])
        self.compress_L2 = nn.Conv2d(128 + 256, 128, kernel_size=1)
        self.compress_L1 = nn.Conv2d(64 + 128, 64, kernel_size=1)
        self.compress_L0 = nn.Conv2d(1 + 64, 32, kernel_size=1)
        # self.compress_L0 = nn.Conv2d(1 + 32, 1, kernel_size=1)

    @staticmethod
    def _max_disp_for_level(level_idx: int, max_disp: int) -> int:
        if level_idx < 2:
            return max_disp
        if level_idx == 2:
            return max_disp - 1
        return max_disp - 2

    def forward(self, ftrs0: list[torch.Tensor], ftrs2: list[torch.Tensor], raw_x0: torch.Tensor, raw_x2: torch.Tensor):
        interp_features_out = []
        warps_x0_out = []
        warps_x2_out = []

        levels = len(self.flow_heads)  # e.g., 4 levels

        # ----------------------------------------------------------------
        # STEP 1: START AT THE BOTTLENECK (Level 4 / Index 3)
        # ----------------------------------------------------------------
        # Resolution: H/8 x W/8, Channels: 256
        interp_ftr, w_x0, w_x2 = self.flow_heads[-1](
            ftrs0[-1], ftrs2[-1], ftrs0[-1], ftrs2[-1]
        )

        # Store Level 4 states
        interp_features_out.append(interp_ftr)
        warps_x0_out.append(w_x0)
        warps_x2_out.append(w_x2)

        # Define our running feature that travels up the network
        running_context_feature = interp_ftr  # Base: [B, 256, H/8, W/8]

        # ----------------------------------------------------------------
        # STEP 2: LOOP UPWARD THROUGH INTERMEDIATE PYRAMIDS (Levels 3 & 2)
        # ----------------------------------------------------------------
        # Corrected step to -1 to move backward cleanly: Index 2 down to Index 1
        for level in range(levels - 2, 0, -1):
            # 1. Upsample the running structural context map from below
            upsampled_context = F.interpolate(
                running_context_feature, scale_factor=2, mode="bilinear", align_corners=True
            )

            # 2. Calculate native high-res flow features for current level
            native_interp, w_x0, w_x2 = self.flow_heads[level](
                ftrs0[level], ftrs2[level], ftrs0[level], ftrs2[level]
            )

            # 3. Concatenate and IMMEDIATELY compress to avoid a heavy model
            fused_ftr = torch.cat([native_interp, upsampled_context], dim=1)
            if level == 2:
                running_context_feature = self.compress_L2(fused_ftr)
            elif level == 1:
                running_context_feature = self.compress_L1(fused_ftr)
            # elif level == 1:
            #     running_context_feature = self.compress_L1(fused_ftr)

            # 4. Track outputs for corresponding decoder stages
            interp_features_out.append(running_context_feature)
            warps_x0_out.append(w_x0)
            warps_x2_out.append(w_x2)

        # ----------------------------------------------------------------
        # STEP 3: TOP-LEVEL FINALIZATION (Level 1 / Index 0)
        # ----------------------------------------------------------------
        # Resolution: H x W, Channels: 1 (Warping raw matrices)
        upsampled_context = F.interpolate(
            running_context_feature, scale_factor=2, mode="bilinear", align_corners=True
        )

        # Calculate full-resolution crisp displacements using Feature Extractor
        # We pass raw input matrices (1 channel) into raw slots!
        native_interp, w_x0, w_x2 = self.flow_heads[0](
            ftrs0[0], ftrs2[0], raw_x0, raw_x2
        )

        # Final Top-Level Channel Fusion
        fused_L0 = torch.cat([native_interp, upsampled_context], dim=1)
        running_context_feature = self.compress_L0(fused_L0)

        interp_features_out.append(running_context_feature)
        warps_x0_out.append(w_x0)
        warps_x2_out.append(w_x2)

        # Reverse lists so they cleanly map from Top (L1) to Bottom (L4) for the Decoder
        return interp_features_out[::-1], warps_x0_out[::-1], warps_x2_out[::-1]
