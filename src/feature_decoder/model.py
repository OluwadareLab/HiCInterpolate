import torch
from typing import List
from torch.nn import Module, Conv2d, ReLU, functional as F, ModuleList
from torch import Tensor


class DiffusionConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding='same',
                 diffusion_steps: int = 2, diffusion_rate: float = 0.15):
        super().__init__()
        self.diffusion_steps = max(0, int(diffusion_steps))
        self.diffusion_rate = float(diffusion_rate)
        self.channel_proj = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding='same')
        self.spatial_proj = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # 4-neighbor discrete Laplacian kernel used for explicit diffusion updates.
        laplacian = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian)

    def _laplacian(self, x: Tensor) -> Tensor:
        kernel = self.laplacian_kernel.expand(x.shape[1], 1, 3, 3).to(dtype=x.dtype)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def forward(self, input: Tensor) -> Tensor:
        x = self.channel_proj(input)
        for _ in range(self.diffusion_steps):
            x = x + self.diffusion_rate * self._laplacian(x)
        return self.spatial_proj(x)


class Block(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 diffusion_steps: int = 2, diffusion_rate: float = 0.15):
        super().__init__()
        self.conv1 = DiffusionConv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     diffusion_steps=diffusion_steps,
                                     diffusion_rate=diffusion_rate)
        self.relu1 = ReLU()
        self.conv2 = DiffusionConv2d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     diffusion_steps=diffusion_steps,
                                     diffusion_rate=diffusion_rate)
        self.relu2 = ReLU()

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        output = self.relu2(x)

        return output


class Fusion(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.convs = ModuleList()
        diffusion_steps = getattr(self.cfg.model, 'diffusion_steps', 2)
        diffusion_rate = getattr(self.cfg.model, 'diffusion_rate', 0.15)
        self.levels = self.cfg.model.fusion_pyramid_level
        init_in_channels = self.cfg.model.init_in_channels
        init_out_channels = self.cfg.model.init_out_channels
        prev_out_channels = 0
        for i in range(self.levels-1):
            m = self.cfg.model.unique_levels
            k = init_out_channels
            out_channels = (k << i) if i < m else (k << m)
            in_channels = (prev_out_channels + out_channels +
                           init_in_channels) * 2 + 4

            convs = ModuleList()
            channels = out_channels*2 if i < m else in_channels
            convs.append(DiffusionConv2d(in_channels=channels,
                                         out_channels=out_channels,
                                         kernel_size=2,
                                         padding='same',
                                         diffusion_steps=diffusion_steps,
                                         diffusion_rate=diffusion_rate))
            channels = in_channels + out_channels
            convs.append(Block(in_channels=channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               diffusion_steps=diffusion_steps,
                               diffusion_rate=diffusion_rate))
            self.convs.append(convs)
            prev_out_channels = prev_out_channels + out_channels

        self.output_conv = DiffusionConv2d(
            in_channels=init_out_channels,
            out_channels=init_in_channels,
            kernel_size=1,
            padding='same',
            diffusion_steps=diffusion_steps,
            diffusion_rate=diffusion_rate)
        self.output_relu = ReLU()

    def forward(self, pyramid: List[Tensor]) -> Tensor:
        if len(pyramid) != self.levels:
            raise ValueError(
                '[ERROR] fusion called with different number of pyramid levels ' f'{len(pyramid)} than it was configured for, {self.levels}.')
        net = pyramid[-1]
        for i in reversed(range(0, self.levels-1)):
            level_size = (pyramid[i].shape)[2:4]
            net = F.interpolate(net, size=level_size, mode='nearest')
            net = self.convs[i][0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = self.convs[i][1](net)
        net = self.output_conv(net)
        net = self.output_relu(net)

        return net
