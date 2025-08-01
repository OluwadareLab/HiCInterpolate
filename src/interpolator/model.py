from ..misc import utils
from torch.nn import Module
from torch import Tensor
from ..fusion import Fusion
from ..flow_estimation import PyramidFlowEstimator
from ..feature_extractor import FeatureExtractor
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Interpolator(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_ext = FeatureExtractor(self.cfg)
        self.flow_est = PyramidFlowEstimator(self.cfg)
        self.fusion = Fusion(self.cfg)

    def forward(self, x0: Tensor, x2: Tensor, time: Tensor) -> Tensor:
        pyramid_levels = self.cfg.model.pyramid_level
        fusion_pyramid_levels = self.cfg.model.fusion_pyramid_level
        if pyramid_levels < fusion_pyramid_levels:
            raise ValueError(
                '[Error] pyramid level must be greater than or equal to fusion level')

        # Create input pyramid
        x0_decoded = x0
        x2_decoded = x2

        pyramid0 = utils.build_image_pyramid(x0_decoded, pyramid_levels)
        pyramid2 = utils.build_image_pyramid(x2_decoded, pyramid_levels)
        image_pyramid = [pyramid0, pyramid2]

        # Feature extractor
        ftr_pyramid0 = self.feature_ext(image_pyramid[0])
        ftr_pyramid2 = self.feature_ext(image_pyramid[1])
        feature_pyramids = [ftr_pyramid0, ftr_pyramid2]

        # Flow estimator
        forward_residual_flow_pyramid = self.flow_est(
            feature_pyramids[0], feature_pyramids[1])
        backward_residual_flow_pyramid = self.flow_est(
            feature_pyramids[1], feature_pyramids[0])

        fusion_pyramid_levels = fusion_pyramid_levels
        forward_flow_pyramid = utils.flow_pyramid_synthesis(
            forward_residual_flow_pyramid)[:fusion_pyramid_levels]
        backward_flow_pyramid = utils.flow_pyramid_synthesis(
            backward_residual_flow_pyramid)[:fusion_pyramid_levels]
        mid_time = torch.ones_like(time, device=time.device) * 0.5
        backward_flow = utils.multiply_pyramid(
            backward_flow_pyramid, mid_time[:, 0])
        forward_flow = utils.multiply_pyramid(
            forward_flow_pyramid, 1-mid_time[:, 0])

        pyramids_to_warp = [utils.concatenate_pyramids(image_pyramid[0][:fusion_pyramid_levels], feature_pyramids[0][:fusion_pyramid_levels]),
                            utils.concatenate_pyramids(image_pyramid[1][:fusion_pyramid_levels], feature_pyramids[1][:fusion_pyramid_levels])]

        forward_warped_pyramid = utils.pyramid_warp(
            pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid = utils.pyramid_warp(
            pyramids_to_warp[1], forward_flow)

        aligned_pyramid = utils.concatenate_pyramids(
            forward_warped_pyramid, backward_warped_pyramid)
        aligned_pyramid = utils.concatenate_pyramids(
            aligned_pyramid, backward_flow)
        aligned_pyramid = utils.concatenate_pyramids(
            aligned_pyramid, forward_flow)

        # Fusion
        prediction = self.fusion(aligned_pyramid)
        return prediction
