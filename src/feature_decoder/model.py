from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SkipFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Localized 1x1 layer to learn a spatial weight mask for every single channel bin
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # Forces weights to stay strictly between 0.0 and 1.0
        )

        # Refinement trunk to process the gated output back to target channel width
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, skip0, skip1):
        # 1. Stack them temporarily to look at both states simultaneously
        stacked = torch.cat([skip0, skip1], dim=1)
        # 2. Generate the dynamic, pixel-wise blending mask (Alpha)
        alpha = self.gate_generator(stacked)
        # 3. Execute the Learnable Blend
        gated_blend = alpha * skip0 + (1.0 - alpha) * skip1
        # 4. Refine and output the clean, bounded target tensor
        return self.refine(gated_blend)


class DecoderBlock(nn.Module):
    def __init__(self, deep_channels, native_channels, out_channels):
        """
        Args:
            deep_channels (int): Channel width of features coming from the bottleneck/lower level (e.g., 256)
            native_channels (int): Channel width of current level skips, flows, and warps (e.g., 128)
            out_channels (int): Target channel output width for this decoder block (e.g., 128)
        """
        super().__init__()

        # Aligns SkipFusion strictly with the native encoder channel sizes, fixing Bug #1
        self.skip_fusion = SkipFusion(channels=native_channels)

        # 1x1 Channel Compression for the upsampled context feature map
        self.has_deep_context = deep_channels > 0
        self.is_fine_level = native_channels <= 32  # Levels 1 & 2 are the "fine" levels with sharper features
        if self.has_deep_context:
            self.upsample_conv = nn.Conv2d(
                in_channels=deep_channels,
                out_channels=native_channels,
                kernel_size=1,
                bias=False
            )
            # Total input width: upsampled_context (native) + unified_skip (native) + interp (native) + w0 (native) + w1 (native)
            total_fused_channels = native_channels * 5
         # No deep context, but we keep the full skip at fine levels to preserve crisp details
        else:
            # At Level 4 (Bottleneck), there is no deep context layer from below
            # Total input width: unified_skip (native) + interp (native) + w0 (native) + w1 (native)
            total_fused_channels = native_channels * 3
        
        if self.has_deep_context and self.is_fine_level:
            total_fused_channels = native_channels * 3 + 2 

        

        # Bottleneck compression layer: IMMEDIATELY squashes channel explosion down to manageable widths
        # This prevents the parameter size of the subsequent 3x3 layers from exploding
        self.channel_squeeze = nn.Conv2d(
            total_fused_channels, out_channels, kernel_size=1, bias=False)

        mid_channels = out_channels // 2

        # Branch A: Preserves sharp, localized boundaries (Loops, TAD Edges)
        self.fine_refinement = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels,
                      kernel_size=1, padding=0, bias=False),  # [CHANGED: k=1]
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Branch B: Preserves broad structural patterns (TAD interiors, Compartments)
        self.structural_refinement = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1,
                      padding=0, bias=False),  # [CHANGED: k=1]
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final aggregation layer for this level
        self.compress = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, interp_ftr, w0, w1, skip0=None, skip2=None, dec_ftr_up=None):

        if self.has_deep_context:
            # 1. Fuse the parallel encoder skips into a single coordinate system
            # Output Shape: [B, native_channels, H, W]
            unified_skip = self.skip_fusion(skip0, skip2)
            # Spatial scale expansion (2x resolution scale-up via smooth bilinear upsampling)
            upsampled_spatial = F.interpolate(
                dec_ftr_up, scale_factor=2, mode="bilinear", align_corners=True
            )
            # Match the target native channel dimension budget
            dec_feat_context = self.upsample_conv(upsampled_spatial)
            fused_inputs = torch.cat([
                dec_feat_context,  # Aligned contextual history
                unified_skip,     # Learnable skip template
                interp_ftr,       # Core flow midpoint features
                w0, w1            # Sharp edge constraints
            ], dim=1)
        else:
            # Execute Bottleneck (Level 4) stacking logic
            fused_inputs = torch.cat([
                interp_ftr,
                w0, w1
            ], dim=1)

        # 3. Apply immediate 1x1 compression to resolve channel explosion before refinement
        squeezed_tensor = self.channel_squeeze(
            fused_inputs)  # Forces width back to out_channels

        # 4. Multi-scale residual processing
        out_fine = self.fine_refinement(squeezed_tensor)
        out_struct = self.structural_refinement(squeezed_tensor)

        # 5. Combine and project back to standard level bounds
        out = torch.cat([out_fine, out_struct], dim=1)
        return self.compress(out)


class FeatureDecoder(nn.Module):
    def __init__(self, cfg, feature_channels=[32, 64, 128, 256], out_channels=1):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = feature_channels

        self.level4 = DecoderBlock(
            deep_channels=0,
            native_channels=self.feature_channels[3],
            out_channels=self.feature_channels[3],
        )

        self.level3 = DecoderBlock(
            deep_channels=self.feature_channels[3],
            native_channels=self.feature_channels[2],
            out_channels=self.feature_channels[2],
        )

        self.level2 = DecoderBlock(
            deep_channels=self.feature_channels[2],
            native_channels=self.feature_channels[1],
            out_channels=self.feature_channels[1],
        )

        self.level1 = DecoderBlock(
            deep_channels=self.feature_channels[1],
            native_channels=self.feature_channels[0],
            out_channels=self.feature_channels[0],
        )

    def forward(self, skips0: List[Tensor], skips2: List[Tensor], interpolations: List[Tensor], warps_0: Tensor, warps_2: Tensor) -> Tensor:
        out = self.level4(interpolations[3], warps_0[3], warps_2[3])
        out = self.level3(
            interpolations[2], warps_0[2], warps_2[2], skips0[2], skips2[2], out)
        out = self.level2(
            interpolations[1], warps_0[1], warps_2[1], skips0[1], skips2[1], out)
        out = self.level1(
            interpolations[0], warps_0[0], warps_2[0], skips0[0], skips2[0], out)
        return out
