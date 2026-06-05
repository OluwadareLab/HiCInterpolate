import os
import sys
import torch
import torch.nn as nn
from src.feature_encoder import FeatureEncoder
from src.flow_predictor import ForwardFlowPredictor, BackwardFlowPredictor
from src.feature_decoder import FeatureDecoder
from torch import Tensor
from src.flow_predictor.model import ForwardFlowPredictor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Interpolator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = FeatureEncoder(self.cfg)
        self.forward_flow = ForwardFlowPredictor(self.cfg)
        self.backward_flow = BackwardFlowPredictor(self.cfg)
        self.feature_decoder = FeatureDecoder(self.cfg)

    @staticmethod
    def concatenate_flow_ftr(ftr_0: list[Tensor], ftr_2: list[Tensor]) -> list[Tensor]:
        mid_ftr = []
        for feature1, feature2 in zip(ftr_0, ftr_2):
            mid_ftr.append(torch.cat([feature1, feature2], dim=1))
        return mid_ftr

    def forward(self, x0: Tensor, x2: Tensor, time: Tensor) -> Tensor:

        # Feature Encoder
        ftr0_stk = self.feature_encoder(x0)
        ftr2_stk = self.feature_encoder(x2)

        # Flow Predictor
        forward_mid_ftr = self.forward_flow(
            ftr0_stk, ftr2_stk, time[:, 0])
        backward_mid_ftr = self.backward_flow(
            ftr2_stk, ftr0_stk, time[:, 0])

        # Feature Alignment
        mid_ftr = self.concatenate_flow_ftr(forward_mid_ftr, backward_mid_ftr)

        # Feature Decoder
        pred = self.feature_decoder(mid_ftr)
        return pred
