from typing import List
from torch.nn import Module, Conv2d, AvgPool2d, Sequential, LeakyReLU, ModuleList
from torch import Tensor
import torch
import config


class SubTreeExtractor(Module):
    def __init__(self):
        super().__init__()
        n = config.EXT_FEATURE_LEVEL
        self.convs = ModuleList()
        in_channels = config.INIT_IN_CHANNELS
        for i in range(n):
            out_channels = config.INIT_OUT_CHANNELS << i
            seq1 = Sequential(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),
                              LeakyReLU(negative_slope=0.2))
            self.convs.append(seq1)
            seq2 = Sequential(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
                              LeakyReLU(negative_slope=0.2))
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
    def __init__(self):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor()

    def forward(self, image_pyramid: List[Tensor]) -> List[Tensor]:
        sub_pyramids = []
        for i in range(len(image_pyramid)):
            capped_sub_levels = min(len(image_pyramid), config.EXT_FEATURE_LEVEL)
            sub_pyramids.append(self.extract_sublevels(
                image_pyramid[i], capped_sub_levels))

        featur_pyramid = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, config.EXT_FEATURE_LEVEL):
                if j <= i:
                    features = torch.cat(
                        [features, sub_pyramids[i-j][j]], axis=1)
            featur_pyramid.append(features)
        return featur_pyramid
