import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm


def gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class PatchDiscriminator(nn.Module):
    """Spectral-norm PatchGAN. Temporally conditioned on the neighbor frames:
    input is concat([x0, frame, x2]) so the critic judges whether `frame`
    is a plausible middle given its neighbors."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            spectral_norm(nn.Conv2d(in_channels, base_channels,
                          kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for i in range(1, n_layers):
            next_ch = min(base_channels * (2 ** i), 512)
            layers += [
                spectral_norm(nn.Conv2d(ch, next_ch, kernel_size=4,
                              stride=2, padding=1, bias=False)),
                gn(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch

        next_ch = min(base_channels * (2 ** n_layers), 512)
        layers += [
            spectral_norm(nn.Conv2d(ch, next_ch, kernel_size=4,
                          stride=1, padding=1, bias=False)),
            gn(next_ch),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(next_ch, 1, kernel_size=4,
                          stride=1, padding=1)),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def d_hinge_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits: Tensor) -> Tensor:
    return -fake_logits.mean()
