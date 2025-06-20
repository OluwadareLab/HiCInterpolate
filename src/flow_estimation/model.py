import torch
import sys
import os
from torch.nn import Module, ModuleList, functional as F
from .block import FlowEstimator
from ..misc import utils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PyramidFlowEstimator(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.levels = ModuleList()
        in_channels = 0
        unique_levels = self.cfg.model.unique_levels
        init_out_channels = self.cfg.model.init_out_channels
        pyramid_level = self.cfg.model.pyramid_level
        num_of_convs = self.cfg.model.flow.num_of_convs
        out_channels = self.cfg.model.flow.out_channels

        for i in range(unique_levels):
            channels = init_out_channels << i
            in_channels = in_channels + channels
            self.levels.append(FlowEstimator(
                num_of_convs=num_of_convs[i], in_channels=in_channels*2, out_channels=out_channels[i]))

        channels = init_out_channels << unique_levels
        in_channels = in_channels + channels
        conv = FlowEstimator(
            num_of_convs=num_of_convs[-1], in_channels=in_channels*2, out_channels=out_channels[-1])
        for i in range(unique_levels, pyramid_level):
            self.levels.append(conv)

    def forward(self, ftr_pyr_a: list[torch.Tensor], ftr_pyr_b: list[torch.Tensor]):
        levels = len(ftr_pyr_a)
        v = self.levels[-1](ftr_pyr_a[-1], ftr_pyr_b[-1])
        residuals = [v]
        for i in reversed(range(0, levels-1)):
            level_size = (ftr_pyr_a[i].shape)[2:4]
            v = F.interpolate(input=2*v, size=level_size)
            warper = utils.warp(ftr_pyr_b[i], v)
            v_residual = self.levels[i](ftr_pyr_a[i], warper)
            residuals.append(v_residual)
            v = v_residual + v

        return list(reversed(residuals))
