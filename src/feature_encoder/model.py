import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InputProjection(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.dilated_expanded = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=6, dilation=2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.local_refine = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.dilated_expanded(x)))
        x = self.act(self.bn2(self.local_refine(x)))
        return x


class DilatedDoubleBlock64To32(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=6,
            dilation=2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.act(x)

        return output


class DilatedDoubleBlock32To16(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=4,
            dilation=2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        return self.pool(x)


class DilatedDoubleBlock16To8(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.pool(x)


class DilatedDoubleBlock8To4(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.pool(x)


class FeatureEncoder(nn.Module):
    def __init__(self, cfg, in_channels=1, out_channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.cfg = cfg

        # 1 > 256, k=7
        self.input_project = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0]//4,
                      kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(out_channels[0]//4),

            nn.Conv2d(in_channels=out_channels[0]//4, out_channels=out_channels[0]//2,
                      kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(out_channels[0]//2),

            nn.Conv2d(in_channels=out_channels[0]//2, out_channels=out_channels[0],
                      kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(out_channels[0]),

            nn.Conv2d(out_channels[0], out_channels[0],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU()
        )

        # 256 > 128 k=7
        self.unique1 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_size=7,
                stride=1,
                padding=6,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[1]),
            nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[1]),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 128 > 128 k=7
        self.shared1 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=7,
                stride=1,
                padding=6,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[1]),
            nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[1]),
            nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 128 > 64 k=5
        self.unique2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[2],
                kernel_size=5,
                stride=1,
                padding=4,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[2]),
            nn.Conv2d(
                in_channels=out_channels[2],
                out_channels=out_channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[2]),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 64 > 64 k=5
        self.shared2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[2],
                out_channels=out_channels[2],
                kernel_size=5,
                stride=1,
                padding=4,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[2]),
            nn.Conv2d(
                in_channels=out_channels[2],
                out_channels=out_channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[2]),
            nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 64 > 32 k=3
        self.unique3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[2],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[3]),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[3]),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 32 > 32 k=3
        self.shared3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[3]),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[3]),
            nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 32 > 16 k=3
        self.unique4 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[4],
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[4]),
            nn.Conv2d(
                in_channels=out_channels[4],
                out_channels=out_channels[4],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[4]),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 16 > 16 k=3
        self.shared4 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[4],
                out_channels=out_channels[4],
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(out_channels[4]),
            nn.Conv2d(
                in_channels=out_channels[4],
                out_channels=out_channels[4],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels[4]),
            nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, ftr: Tensor) -> List[Tensor]:
        proj = self.input_project(ftr)
        unique1 = self.unique1(proj)
        shared1 = self.shared1(unique1)
        unique2 = self.unique2(shared1)
        shared2 = self.shared2(unique2)
        unique3 = self.unique3(shared2)
        shared3 = self.shared3(unique3)
        unique4 = self.unique4(shared3)
        shared4 = self.shared4(unique4)
        return [proj, unique1, shared1, unique2, shared2, unique3, shared3, unique4, shared4]
