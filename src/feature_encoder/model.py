from typing import List
import torch.nn as nn
from torch import Tensor


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample: bool):
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.encoder(x)
        latent = self.downsample(skip)
        return skip, latent


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: List[int] = None):
        super().__init__()
        self.out_channels = out_channels or [16, 32, 64, 128]
        kernels = [3, 3, 3, 3]

        blocks = []
        prev_channels = in_channels
        for idx, (channels, kernel) in enumerate(zip(self.out_channels, kernels)):
            blocks.append(EncoderBlock(
                prev_channels, channels, kernel_size=kernel,
                downsample=idx < len(self.out_channels) - 1,
            ))
            prev_channels = channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> tuple[List[Tensor], Tensor]:
        """Encode input and return skip connections plus the bottleneck latent.

        Returns:
            skips:     List of pre-pool feature maps at each encoder level.
                       skips[0] is finest [B, 256, H, W],
                       skips[-1] is coarsest skip [B, 32, H/8, W/8].
            bottleneck: Final post-pool latent [B, 32, H/16, W/16],
                       capturing the most compressed global representation.
        """
        skips = []
        latent = x
        for block in self.blocks:
            skip, latent = block(latent)
            skips.append(skip)
        # latent here is the final MaxPool output — the true bottleneck
        return skips
