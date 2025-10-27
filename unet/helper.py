import os
import sys
import torch.nn.functional as F
import torch.nn as nn
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        res = self.double_conv(img)
        return res


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, img):
        res = self.maxpool(img)
        return res


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels=in_channels, out_channels=out_channels)

    def forward(self, img, res_img):
        img = self.up(img)
        diff_y = res_img.size()[2] - img.size()[2]
        diff_x = res_img.size()[3] - img.size()[3]

        img = F.pad(img, [diff_x//2, diff_x-diff_x //
                    2, diff_y//2, diff_y-diff_y//2])
        cat_img = torch.cat([img, res_img], dim=1)
        res = self.conv(cat_img)
        return res


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels, kernel_size=1)

    def forward(self, img):
        res = self.conv(img)
        return res
