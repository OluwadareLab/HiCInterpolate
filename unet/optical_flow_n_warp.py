'''
https://github.com/Utkal97/Object-Tracking/blob/main/LucasKanadeOptFlow.py
https://www.youtube.com/watch?v=VSSyPskheaE
'''

import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def optical_flow(img1: Tensor, img3: Tensor, window_size=3, eps=1e-6):
    assert img1.shape == img3.shape
    b, c, h, w = img1.shape
    assert c == 1, "Support only singel channel image"

    kernel_x = torch.tensor([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=torch.float, device=img1.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -1, -1],
                            [0,  0,  0],
                            [1,  1,  1]], dtype=torch.float, device=img1.device).view(1, 1, 3, 3)
    kernel_t = torch.ones((1, 1, 3, 3), dtype=torch.float, device=img1.device)

    fx = F.conv2d(img1, kernel_x, padding=1)
    fy = F.conv2d(img1, kernel_y, padding=1)
    ft = F.conv2d(img3, kernel_t, padding=1) - \
        F.conv2d(img1, kernel_t, padding=1)

    win_size = window_size
    weight = torch.ones((1, 1, win_size, win_size),
                        dtype=torch.float, device=img1.device)

    fx2 = F.conv2d(fx * fx, weight, padding=win_size//2)
    fy2 = F.conv2d(fy * fy, weight, padding=win_size//2)
    fxy = F.conv2d(fx * fy, weight, padding=win_size//2)
    fxt = F.conv2d(fx * ft, weight, padding=win_size//2)
    fyt = F.conv2d(fy * ft, weight, padding=win_size//2)

    det = (fx2 * fy2 - fxy ** 2) + eps
    u = (-fy2 * fxt + fxy * fyt) / det
    v = (fxy * fxt - fx2 * fyt) / det

    return u, v


def warp_image(img, flow):
    b, c, h, w = img.shape

    yy, xx = torch.meshgrid(
        torch.arange(0, h, device=img.device),
        torch.arange(0, w, device=img.device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=0).float()
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    grid = torch.add(grid, flow)
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(w - 1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(h - 1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)

    warped = F.grid_sample(
        img, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    return warped
