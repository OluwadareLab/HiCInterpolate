from typing import List
from torch.nn import Module, Conv2d, AvgPool2d, Sequential, ReLU, ModuleList, functional as F
from torch import Tensor
import torch


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


class SubTreeExtractor(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n = self.cfg.model.ext_feature_level
        diffusion_steps = getattr(self.cfg.model, 'diffusion_steps', 2)
        diffusion_rate = getattr(self.cfg.model, 'diffusion_rate', 0.15)
        self.convs = ModuleList()
        in_channels = self.cfg.model.init_in_channels
        for i in range(n):
            out_channels = self.cfg.model.init_out_channels << i
            seq1 = Sequential(DiffusionConv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=3,
                                              padding='same',
                                              diffusion_steps=diffusion_steps,
                                              diffusion_rate=diffusion_rate),
                              ReLU())
            self.convs.append(seq1)
            seq2 = Sequential(DiffusionConv2d(in_channels=out_channels,
                                              out_channels=out_channels,
                                              kernel_size=3,
                                              padding='same',
                                              diffusion_steps=diffusion_steps,
                                              diffusion_rate=diffusion_rate),
                              ReLU())
            self.convs.append(seq2)
            in_channels = out_channels

        self.avgpool = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, image: Tensor, n: int) -> List[Tensor]:
        head = image
        pyramid = []
        for i in range(n):
            head = self.convs[2*i](head)
            head = self.convs[2*i+1](head)
            pyramid.append(head)
            if i < n-1:
                head = self.avgpool(head)
        return pyramid


class FeatureExtractor(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.extract_sublevels = SubTreeExtractor(self.cfg)

    def forward(self, image_pyramid: List[Tensor]) -> List[Tensor]:
        sub_pyramids = []
        ext_feature_level = self.cfg.model.ext_feature_level
        for i in range(len(image_pyramid)):
            capped_sub_levels = min(len(image_pyramid), ext_feature_level)
            sub_pyramids.append(self.extract_sublevels(
                image_pyramid[i], capped_sub_levels))

        featur_pyramid = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, ext_feature_level):
                if j <= i:
                    features = torch.cat(
                        [features, sub_pyramids[i-j][j]], axis=1)
            featur_pyramid.append(features)
        return featur_pyramid
