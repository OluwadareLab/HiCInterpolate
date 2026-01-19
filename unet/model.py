from helper import *
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import optical_flow_n_warp as ofw
from torchvision.models.optical_flow import raft_large
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FeatureCorrelationFlow(nn.Module):
    def __init__(self, radius=4, hidden_dim=64):
        """
        radius: size of local neighborhood (correlation window = (2r+1)^2)
        hidden_dim: number of channels in refinement CNN
        """
        super().__init__()
        self.radius = radius

        corr_channels = (2 * radius + 1) ** 2
        self.refine_net = nn.Sequential(
            nn.Conv2d(corr_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1)  # output flow (u,v)
        )

    def compute_local_correlation(self, F1, F2):
        """Compute local correlation volume between F1 and shifted F2."""
        B, C, H, W = F1.shape
        corr_tensors = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                shifted = torch.roll(F2, shifts=(dy, dx), dims=(2, 3))
                corr = (F1 * shifted).sum(1, keepdim=True) / C**0.5
                corr_tensors.append(corr)
        corr_volume = torch.cat(corr_tensors, dim=1)
        return corr_volume  # [B, (2r+1)^2, H, W]

    def forward(self, F1, F2):
        corr = self.compute_local_correlation(F1, F2)
        flow = self.refine_net(corr)
        return flow  # [B, 2, H, W]


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024, bilinear=False):
        super(Encoder, self).__init__()
        self.window_size = 5
        self.eps = 0.05
        self.time = 0.5
        self.inc = (DoubleConv(in_channels=in_channels, out_channels=64))
        self.down1 = (Down(in_channels=64, out_channels=128))
        self.down2 = (Down(in_channels=128, out_channels=256))
        self.down3 = (Down(in_channels=256, out_channels=512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(in_channels=512, out_channels=out_channels//factor))

        self.flow = FeatureCorrelationFlow(radius=5)

    def forward(self, img):
        x1 = self.inc(img)  # 256
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 64
        x4 = self.down3(x3)  # 32
        x5 = self.down4(x4)  # 16

        return x1, x2, x3, x4, x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)


class Decoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1, bilinear=False):
        super(Decoder, self).__init__()

        factor = 2 if bilinear else 1
        self.up1 = (Up(in_channels=in_channels, out_channels=512 //
                    factor, bilinear=bilinear))
        self.up2 = (Up(in_channels=512, out_channels=256 //
                    factor, bilinear=bilinear))
        self.up3 = (Up(in_channels=256, out_channels=128 //
                    factor, bilinear=bilinear))
        self.up4 = (Up(in_channels=128, out_channels=64, bilinear=bilinear))
        self.outc = (OutConv(in_channels=64, out_channels=out_channels))

    def forward(self, x5, x4, x3, x2, x1):
        res = self.up1(x5, x4)  # 32
        res = self.up2(res, x3)  # 64
        res = self.up3(res, x2)  # 128
        res = self.up4(res, x1)  # 256
        res = self.outc(res)    # 256
        return res

    def use_checkpointing(self):
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.encoder = Encoder(in_channels=self.in_channels,
                               out_channels=1024, bilinear=self.bilinear)
        self.decoder = Decoder(
            in_channels=1024, out_channels=self.out_channels, bilinear=self.bilinear)

    def _get_flow_img(self, img1, img2):
        fwd_u, fwd_v = ofw.optical_flow(img1, img2)
        fwd_flow = torch.cat((0.5*fwd_u, 0.5*fwd_v), dim=1)
        fwd_img = ofw.warp_image(img1, fwd_flow)

        bwd_u, bwd_v = ofw.optical_flow(img2, img1)
        bwd_flow = torch.cat((0.5*bwd_u, 0.5*bwd_v), dim=1)
        bwd_img = ofw.warp_image(img2, bwd_flow)

        img2_pred = 0.5 * torch.add(fwd_img, bwd_img)
        return img2_pred

    def forward(self, img1, img3):
        img2 = self._get_flow_img(img1, img3)
        x1, x2, x3, x4, x5 = self.encoder(img2)
        del img3
        res = self.decoder(x5, x4, x3, x2, x1)
        del x1, x2, x3, x4, x5,
        return res

    def use_checkpointing(self):
        self.encoder = torch.utils.checkpoint(self.encoder)
        self.decoder = torch.utils.checkpoint(self.decoder)
