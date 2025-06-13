from typing import List
from torch.nn import functional as F, AvgPool2d
import torch
import config


def build_image_pyramid(image: torch.Tensor) -> List[torch.Tensor]:
    levels = config.PYRAMID_LEVEL
    pyramid = []
    pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
    for i in range(0, levels):
        pyramid.append(image)
        if i < levels-1:
            image = pool(image)

    return pyramid


def warp(image: torch.Tensor, flow: torch.Tensor):
    B, _, H, W = image.size()

    y, x = torch.meshgrid(
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=-1).float()
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    flow = flow.permute(0, 2, 3, 1)
    flowed_grid = grid + flow
    flowed_grid_x = 2.0 * flowed_grid[..., 0] / (W - 1) - 1.0
    flowed_grid_y = 2.0 * flowed_grid[..., 1] / (H - 1) - 1.0
    normalized_grid = torch.stack((flowed_grid_x, flowed_grid_y), dim=-1)
    warped = F.grid_sample(image, normalized_grid, mode='bilinear',
                           padding_mode='border', align_corners=True)

    return warped


def flow_pyramid_synthesis(residual_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    flow = residual_pyramid[-1]
    flow_pyramid = [flow]

    for residual_flow in reversed(residual_pyramid[:-1]):
        level_size = (residual_flow.shape)[2:4]
        flow = F.interpolate(2 * flow, size=level_size,
                             mode='bilinear', align_corners=True)
        flow = residual_flow + flow
        flow_pyramid.append(flow)

    return list(reversed(flow_pyramid))


def multiply_pyramid(pyramid: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
    results = []
    scl = scalar.view(-1, 1, 1, 1)
    for image in pyramid:
        res = image * scl
        results.append(res)

    return results


def pyramid_warp(feature_pyramid: List[torch.Tensor], flow_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(image=features, flow=flow))

    return warped_feature_pyramid


def concatenate_pyramids(pyramid1: List[torch.Tensor], pyramid2: List[torch.Tensor]) -> List[torch.Tensor]:
    result = []
    for feature1, feature2 in zip(pyramid1, pyramid2):
        result.append(torch.cat([feature1, feature2], dim=1))
    return result
