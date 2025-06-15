from torch.nn import Module, ModuleList, functional as F
from .block import FlowEstimator
from utils import warp
import torch
import config

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PyramidFlowEstimator(Module):
    def __init__(self):
        super().__init__()
        self.levels = ModuleList()

        in_channels = 0
        for i in range(config.UNIQUE_LEVELS):
            channels = config.INIT_OUT_CHANNELS << i
            in_channels = in_channels + channels
            self.levels.append(FlowEstimator(
                num_of_convs=config.FLOW_NUM_OF_CONVS[i], in_channels=in_channels*2, out_channels=config.FLOW_OUT_CHANNELS[i]))

        channels = config.INIT_OUT_CHANNELS << config.UNIQUE_LEVELS
        in_channels = in_channels + channels
        conv = FlowEstimator(
            num_of_convs=config.FLOW_NUM_OF_CONVS[-1], in_channels=in_channels*2, out_channels=config.FLOW_OUT_CHANNELS[-1])
        for i in range(config.UNIQUE_LEVELS, config.PYRAMID_LEVEL):
            self.levels.append(conv)

    def forward(self, ftr_pyr_a: list[torch.Tensor], ftr_pyr_b: list[torch.Tensor]):
        levels = len(ftr_pyr_a)
        v = self.levels[-1](ftr_pyr_a[-1], ftr_pyr_b[-1])
        residuals = [v]
        for i in reversed(range(0, levels-1)):
            level_size = (ftr_pyr_a[i].shape)[2:4]
            v = F.interpolate(input=2*v, size=level_size)
            warper = warp(ftr_pyr_b[i], v)
            v_residual = self.levels[i](ftr_pyr_a[i], warper)
            residuals.append(v_residual)
            v = v_residual + v

        return list(reversed(residuals))