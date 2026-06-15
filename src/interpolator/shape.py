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
    def __init__(self, base_channels=64, out_channels=1):
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
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

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

    def forward(self, ftr0: Tensor, ftr2: Tensor, base_flow: Tensor = None, 
                warp_ftr0: Tensor = None, warp_ftr2: Tensor = None) -> tuple[Tensor, Tensor]:
        flow = self.flow_estimator(self.cost_volume(ftr0, ftr2))
        if base_flow is not None:
            flow = flow + base_flow
            
        if warp_ftr0 is None: warp_ftr0 = ftr0
        if warp_ftr2 is None: warp_ftr2 = ftr2
            
        grid0 = self.flow_to_grid(0.5 * flow)
        grid2 = self.flow_to_grid(-0.5 * flow)
        warped0 = F.grid_sample(
            warp_ftr0, grid0, mode="bilinear", padding_mode="border", align_corners=True)
        warped2 = F.grid_sample(
            warp_ftr2, grid2, mode="bilinear", padding_mode="border", align_corners=True)
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

    def forward(self, latent0: Tensor, latent2: Tensor, skips0: List[Tensor] = None, skips2: List[Tensor] = None,
                warp_latent0: Tensor = None, warp_latent2: Tensor = None, 
                warp_skips0: List[Tensor] = None, warp_skips2: List[Tensor] = None):
        coarse_flow = None
        latent_interp, flow = self.flow_heads[3](latent0, latent2, coarse_flow, warp_latent0, warp_latent2)

        coarse_flow = self._upsample_flow(flow, skips0[2].shape[-2:])
        interp3, flow = self.flow_heads[2](skips0[2], skips2[2], coarse_flow, 
                                           warp_skips0[2] if warp_skips0 else None, 
                                           warp_skips2[2] if warp_skips2 else None)

        coarse_flow = self._upsample_flow(flow, skips0[1].shape[-2:])
        interp2, flow = self.flow_heads[1](skips0[1], skips2[1], coarse_flow,
                                           warp_skips0[1] if warp_skips0 else None, 
                                           warp_skips2[1] if warp_skips2 else None)
                                           
        coarse_flow = self._upsample_flow(flow, skips0[0].shape[-2:])
        interp1, flow = self.flow_heads[0](skips0[0], skips2[0], coarse_flow,
                                           warp_skips0[0] if warp_skips0 else None, 
                                           warp_skips2[0] if warp_skips2 else None)

        return latent_interp, [interp1, interp2, interp3]


def erosion2d(x: Tensor, kernel_size: int = 3) -> Tensor:
    padding = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size, stride=1, padding=padding)


def dilation2d(x: Tensor, kernel_size: int = 3) -> Tensor:
    padding = kernel_size // 2
    return F.max_pool2d(x, kernel_size, stride=1, padding=padding)


def opening2d(x: Tensor, kernel_size: int = 3) -> Tensor:
    return dilation2d(erosion2d(x, kernel_size), kernel_size)


def _make_gaussian_kernel(kernel_size: int = 7, sigma: float = 1.5) -> Tensor:
    """Return a normalized 2-D Gaussian kernel as a (1,1,k,k) tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return (g / g.sum()).view(1, 1, kernel_size, kernel_size)


class HiCFeatureExtractorNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.struct_encoder = EncoderHead(in_channels, base_channels)
        self.noise_encoder = EncoderHead(in_channels, base_channels)
        self.flow_predictor = FlowPredictor(cfg=None, feature_channels=[
                                             base_channels, base_channels * 2, base_channels * 4, base_channels * 8], max_disp=4)
        self.struct_decoder = DecoderHead(base_channels, out_channels=1)
        self.dots_decoder = DecoderHead(base_channels, out_channels=2)

        # Fixed Gaussian blur for structure/dot separation.
        # opening2d (erode→dilate) was destroying Hi-C peaks; Gaussian blur
        # correctly preserves the smooth TAD-scale background while keeping
        # sharp loop dots in the residual.
        _kernel = _make_gaussian_kernel(kernel_size=7, sigma=1.5)
        self.register_buffer("_blur_kernel", _kernel)

    def _gaussian_blur(self, x: Tensor) -> Tensor:
        """Apply fixed Gaussian blur channel-wise."""
        b, c, h, w = x.shape
        k = self._blur_kernel.expand(c, 1, -1, -1)  # (C,1,k,k)
        pad = self._blur_kernel.shape[-1] // 2
        return F.conv2d(x, k, padding=pad, groups=c)

    def forward(self, x0, x2):
        # Gaussian Structure/Dot Separation
        # A Gaussian blur correctly extracts the smooth TAD-scale background
        # while leaving sharp loop dots in the residual.  Morphological opening
        # (erode→dilate) was destroying peaks on the sparse Hi-C matrix.
        s0 = self._gaussian_blur(x0).clamp(0.0, 1.0)   # smooth structural background
        n0 = (x0 - s0).clamp(min=0.0)                  # sparse positive residual (dots)

        s2 = self._gaussian_blur(x2).clamp(0.0, 1.0)
        n2 = (x2 - s2).clamp(min=0.0)

        # Encode structure for flow estimation
        latent_s0, skips_s0 = self.struct_encoder(s0)
        latent_s2, skips_s2 = self.struct_encoder(s2)

        # Encode dots/residual to be warped by the structural flow
        latent_n0, skips_n0 = self.noise_encoder(n0)
        latent_n2, skips_n2 = self.noise_encoder(n2)

        # Predict flow from structure, warp structure features
        latent_interp_s, skips_interp_s = self.flow_predictor(
            latent_s0, latent_s2, skips_s0, skips_s2,
            latent_s0, latent_s2, skips_s0, skips_s2)

        # Predict flow from structure, warp dot features
        latent_interp_n, skips_interp_n = self.flow_predictor(
            latent_s0, latent_s2, skips_s0, skips_s2,
            latent_n0, latent_n2, skips_n0, skips_n2)

        # Decode warped dots and contact mask
        dots_out = self.dots_decoder(latent_interp_n, skips_interp_n)
        dots_logits, mask_logits = dots_out.chunk(2, dim=1)

        pred_dots = torch.sigmoid(dots_logits)
        pred_mask = torch.sigmoid(mask_logits)

        # Gated combination: only predict non-zero where mask fires.
        # Dropping the additive pred_struct floor lets the model output true
        # zeros, which is essential for sparsity preservation in Hi-C.
        combined_prob = (pred_dots * pred_mask).clamp(min=1e-6, max=1.0 - 1e-6)

        # Convert back to logits so that torch.sigmoid(pred_logits) == combined_prob
        pred_logits = torch.log(combined_prob / (1.0 - combined_prob))

        return pred_logits, pred_mask
