from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderHead(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

    def forward(self, x0):
        x1 = self.inc(x0)
        p1 = self.pool1(x1)
        x2 = self.down1(p1)
        p2 = self.pool2(x2)
        x3 = self.down2(p2)
        p3 = self.pool3(x3)

        latent = self.bottleneck(p3)
        skips = [x1, x2, x3]
        return latent, skips


class DecoderHead(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up3 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 2, base_channels)
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, bottleneck, skips):
        x1, x2, x3 = skips

        x = self.up1(bottleneck)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return self.final_conv(x)


class FlowEstimationBlock(nn.Module):
    def __init__(self, feature_channels: int, max_disp: int = 4):
        super().__init__()
        self.max_disp = max_disp
        self.search_range = 2 * max_disp + 1
        cost_channels = self.search_range ** 2

        self.flow_estimator = nn.Sequential(
            nn.Conv2d(cost_channels, feature_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, max(16, feature_channels // 2),
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(16, feature_channels // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(16, feature_channels // 2), 2, kernel_size=1),
        )
        self.blend_mask = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + 2, feature_channels,
                      kernel_size=1, bias=False),
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

    def forward(self, ftr0: Tensor, ftr2: Tensor, base_flow: Tensor = None) -> tuple[Tensor, Tensor]:
        flow = self.flow_estimator(self.cost_volume(ftr0, ftr2))
        if base_flow is not None:
            flow = flow + base_flow
        grid0 = self.flow_to_grid(0.5 * flow)
        grid2 = self.flow_to_grid(-0.5 * flow)
        warped0 = F.grid_sample(
            ftr0, grid0, mode="bilinear", padding_mode="border", align_corners=True)
        warped2 = F.grid_sample(
            ftr2, grid2, mode="bilinear", padding_mode="border", align_corners=True)
        alpha = self.blend_mask(torch.cat([warped0, warped2, flow], dim=1))
        interpolation = alpha * warped0 + (1.0 - alpha) * warped2
        return interpolation, flow


class FlowPredictor(nn.Module):
    def __init__(self, cfg, feature_channels: List[int] = None, max_disp: int = 4):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = feature_channels or [64, 128, 256, 512]
        self.flow_heads = nn.ModuleList([
            FlowEstimationBlock(channels, max_disp=max(1, max_disp - idx))
            for idx, channels in enumerate(self.feature_channels)
        ])

    @staticmethod
    def _upsample_flow(flow: Tensor, size: tuple[int, int]) -> Tensor:
        _, _, h_old, w_old = flow.shape
        h_new, w_new = size
        flow = F.interpolate(
            flow, size=size, mode="bilinear", align_corners=True)
        flow[:, 0] *= w_new / max(w_old, 1)
        flow[:, 1] *= h_new / max(h_old, 1)
        return flow

    def forward(self, latent0: Tensor, latent2: Tensor, skips0: List[Tensor] = None, skips2: List[Tensor] = None):
        coarse_flow = None
        latent_interp, flow = self.flow_heads[3](latent0, latent2, coarse_flow)

        coarse_flow = self._upsample_flow(flow, skips0[2].shape[-2:])
        interp3, flow = self.flow_heads[2](skips0[2], skips2[2], coarse_flow)

        coarse_flow = self._upsample_flow(flow, skips0[1].shape[-2:])
        interp2, flow = self.flow_heads[1](skips0[1], skips2[1], coarse_flow)
        coarse_flow = self._upsample_flow(flow, skips0[0].shape[-2:])
        interp1, flow = self.flow_heads[0](skips0[0], skips2[0], coarse_flow)

        return latent_interp, [interp1, interp2, interp3]


class HiCFeatureExtractorNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.encoder = EncoderHead(in_channels, base_channels)
        self.flow_predictor = FlowPredictor(cfg=None, feature_channels=[
                                            base_channels, base_channels * 2, base_channels * 4, base_channels * 8], max_disp=4)
        self.dots_decoder = DecoderHead(base_channels)
        # self.horiz_decoder = DecoderHead(base_channels)
        # self.vert_decoder = DecoderHead(base_channels)

    def forward(self, x0, x2):
        latent0, skips0 = self.encoder(x0)
        latent2, skips2 = self.encoder(x2)

        latent_interp, skips_interp = self.flow_predictor(
            latent0, latent2, skips0, skips2)

        out_dots = self.dots_decoder(latent_interp, skips_interp)
        # out_horiz = self.horiz_decoder(latent_interp, skips_interp)
        # out_vert = self.vert_decoder(latent_interp, skips_interp)

        return out_dots  # out_horiz, pred_vert
