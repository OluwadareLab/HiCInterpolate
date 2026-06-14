from typing import List
from torch.nn import functional as F, AvgPool2d
import torch
from skimage import morphology, filters
from scipy.signal import find_peaks
import cupyx.scipy.signal as cu_signal
from cupyx.scipy import ndimage as cu_ndimage
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

if not hasattr(cp, 'float32'):
    cp.float32 = np.float32
    cp.float64 = np.float64
    cp.int32 = np.int32
    cp.int64 = np.int64


def build_image_pyramid(image: torch.Tensor, levels: int) -> List[torch.Tensor]:
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
                           padding_mode='reflection', align_corners=True)

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


def concatenate_pyramids(pyramid1: List[torch.Tensor], pyramid2: List[torch.Tensor]) -> List[torch.Tensor]:
    result = []
    for feature1, feature2 in zip(pyramid1, pyramid2):
        result.append(torch.cat([feature1, feature2], dim=1))
    return result


def pyramid_warp(feature_pyramid: List[torch.Tensor], flow_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(image=features, flow=flow))

    return warped_feature_pyramid


def extract_diagonal_squares(matrix, window_size=5):
    n = matrix.shape[0]
    insulation_score = np.zeros(n)
    for i in range(window_size, n - window_size):
        local_box = matrix[i-window_size:i, i:i+window_size]
        insulation_score[i] = np.mean(local_box)

    inverted_score = -insulation_score
    boundaries, _ = find_peaks(inverted_score, prominence=0.05)

    diag_squares_matrix = np.zeros_like(matrix)

    all_bounds = [0] + list(boundaries) + [n]
    for start, end in zip(all_bounds[:-1], all_bounds[1:]):
        diag_squares_matrix[start:end,
                            start:end] = matrix[start:end, start:end]

    return diag_squares_matrix


def extract_diagonal_squares_cuda_approx(matrix, window_size=5, prominence=0.05):
    if not torch.is_tensor(matrix):
        matrix = torch.as_tensor(matrix)

    orig_shape = matrix.shape
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).unsqueeze(0)

    B, C, H, W = matrix.shape
    N = B * C
    matrix_flat = matrix.view(N, H, W)

    # Vectorized insulation score
    # insulation_score[i] = mean(matrix[i-w:i, i:i+w])
    # This is a convolution with a window_size x window_size kernel shifted from diagonal
    diag_squares_matrix = torch.zeros_like(matrix_flat)

    for i in range(N):
        m = matrix_flat[i]
        insulation_score = torch.zeros(H, device=m.device)
        # Still loop over H for insulation score as it's complex to vectorize fully with shifting window
        for j in range(window_size, H - window_size):
            local_box = m[j-window_size:j, j:j+window_size]
            insulation_score[j] = local_box.mean()

        inverted_score = -insulation_score
        cp_inverted_score = cp.from_dlpack(inverted_score.contiguous())
        boundaries, _ = cu_signal.find_peaks(
            cp_inverted_score, prominence=prominence)

        torch_boundaries = torch.from_dlpack(boundaries)
        all_bounds = torch.cat([
            torch.zeros(1, device=m.device, dtype=torch.long),
            torch_boundaries.to(torch.long),
            torch.tensor([H], device=m.device, dtype=torch.long),
        ])
        for start, end in zip(all_bounds[:-1].tolist(), all_bounds[1:].tolist()):
            diag_squares_matrix[i, start:end,
                                start:end] = m[start:end, start:end]

    return diag_squares_matrix.view(orig_shape)


def get_exclusion_mask(diagonal_squares, dilation_size=3):
    diagonal_mask = diagonal_squares > 0.05
    exclusion_mask = morphology.binary_dilation(
        diagonal_mask, morphology.square(dilation_size))
    return ~exclusion_mask


def get_exclusion_mask_cuda_approx(diagonal_squares, dilation_size=3):
    orig_shape = diagonal_squares.shape
    if diagonal_squares.ndim == 2:
        diagonal_squares = diagonal_squares.unsqueeze(0).unsqueeze(0)

    B, C, H, W = diagonal_squares.shape
    diagonal_mask = diagonal_squares > 0.05

    cupy_diagonal_mask = cp.from_dlpack(
        diagonal_mask.view(-1, H, W).to(torch.uint8).contiguous())

    footprint = cp.ones((1, dilation_size, dilation_size), dtype=cp.uint8)
    cupy_exclusion_mask = cu_ndimage.binary_dilation(
        cupy_diagonal_mask,
        structure=footprint
    )

    exclusion_mask = torch.from_dlpack(cupy_exclusion_mask).to(torch.bool)
    return (~exclusion_mask).view(orig_shape)


def image_segmentation_cuda_approx(hic_matrix):
    if not torch.is_tensor(hic_matrix):
        hic_matrix = torch.as_tensor(hic_matrix)

    orig_shape = hic_matrix.shape
    if hic_matrix.ndim == 2:
        hic_matrix = hic_matrix.unsqueeze(0).unsqueeze(0)

    B, C, H, W = hic_matrix.shape

    hic_flat = hic_matrix.view(B*C, -1)
    q = torch.quantile(hic_flat, 0.001, dim=1, keepdim=True)
    background = (hic_flat < q).view(B, C, H, W)

    background_mask = get_exclusion_mask_cuda_approx(
        background, dilation_size=1)

    diagonal_dots = extract_diagonal_squares_cuda_approx(
        hic_matrix * background_mask, window_size=1)
    diagonal_mask = get_exclusion_mask_cuda_approx(
        diagonal_dots, dilation_size=1)

    cp_hic = cp.from_dlpack(hic_matrix.view(-1, H, W).contiguous())
    cp_diag_mask = cp.from_dlpack(
        diagonal_mask.view(-1, H, W).to(torch.uint8).contiguous())

    disk1_2d = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.uint8)
    footprint = disk1_2d[cp.newaxis, :, :]
    cp_dots = cu_ndimage.white_tophat(
        cp_hic, footprint=footprint) * cp_diag_mask
    dots = torch.from_dlpack(cp_dots).view(B, C, H, W)

    cp_h_edges = cp.abs(cu_ndimage.sobel(cp_hic, axis=1)) * cp_diag_mask
    horizontal_edges = torch.from_dlpack(cp_h_edges).view(B, C, H, W) / 8.0

    cp_v_edges = cp.abs(cu_ndimage.sobel(cp_hic, axis=2)) * cp_diag_mask
    vertical_edges = torch.from_dlpack(cp_v_edges).view(B, C, H, W) / 8.0

    if len(orig_shape) == 2:
        return diagonal_dots.squeeze(0).squeeze(0), dots.squeeze(0).squeeze(0), \
            horizontal_edges.squeeze(0).squeeze(
                0), vertical_edges.squeeze(0).squeeze(0)

    return diagonal_dots, dots, horizontal_edges, vertical_edges


def image_segmentation_batch(hic_matrix):
    is_torch = torch.is_tensor(hic_matrix)
    if is_torch:
        device = hic_matrix.device
        hic_matrix_np = hic_matrix.detach().cpu().numpy()
    else:
        hic_matrix_np = np.array(hic_matrix)

    orig_shape = hic_matrix_np.shape
    if hic_matrix_np.ndim == 2:
        hic_matrix_np = hic_matrix_np[np.newaxis, np.newaxis, :, :]

    B, C, _,  _ = hic_matrix_np.shape

    out_squares = np.zeros_like(hic_matrix_np)
    out_dots = np.zeros_like(hic_matrix_np)
    out_h_edges = np.zeros_like(hic_matrix_np)
    out_v_edges = np.zeros_like(hic_matrix_np)

    for b in range(B):
        for c in range(C):
            sample = hic_matrix_np[b, c]
            background = sample < np.percentile(sample, 0.1)
            background_mask = get_exclusion_mask(background, dilation_size=1)

            diagonal_dots = extract_diagonal_squares(
                sample * background_mask, window_size=1)
            diagonal_mask = get_exclusion_mask(diagonal_dots, dilation_size=1)

            dot_size = 1
            dots = morphology.white_tophat(
                sample, morphology.disk(dot_size)) * diagonal_mask
            h_edges = np.abs(filters.sobel_h(sample)) * diagonal_mask
            v_edges = np.abs(filters.sobel_v(sample)) * diagonal_mask

            diagonal_squares = diagonal_dots

            out_squares[b, c] = diagonal_squares
            out_dots[b, c] = dots
            out_h_edges[b, c] = h_edges
            out_v_edges[b, c] = v_edges

    results = [out_squares, out_dots, out_h_edges, out_v_edges]
    if is_torch:
        results = [torch.from_numpy(res).to(device) for res in results]

    if len(orig_shape) == 2:
        results = [res.squeeze(0).squeeze(
            0) if is_torch else res[0, 0] for res in results]

    return tuple(results)


def image_segmentaion(hic_matrix, filename=None):
    background = hic_matrix < np.percentile(hic_matrix, 0.1)
    background_mask = get_exclusion_mask(background, dilation_size=1)

    diagonal_dots = extract_diagonal_squares(
        hic_matrix * background_mask, window_size=1)
    diagonal_mask = get_exclusion_mask(diagonal_dots, dilation_size=1)

    dot_size = 1
    dots = morphology.white_tophat(
        hic_matrix, morphology.disk(dot_size)) * diagonal_mask
    horizontal_edges = np.abs(filters.sobel_h(hic_matrix)) * diagonal_mask
    vertical_edges = np.abs(filters.sobel_v(hic_matrix)) * diagonal_mask

    if filename is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax = axes.ravel()

        opts_matrix = dict(cmap='YlOrRd', origin='lower')
        opts_edges = dict(cmap='Reds', origin='lower')

        im0 = ax[0].imshow(hic_matrix, **opts_matrix)
        ax[0].set_title('Original Normalized Hi-C Map')
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im2 = ax[1].imshow(dots, **opts_edges)
        ax[1].set_title('Point Enrichments')
        fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        im3 = ax[2].imshow(horizontal_edges, **opts_edges)
        ax[2].set_title('Horizontal Gradients')
        fig.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)

        im4 = ax[3].imshow(vertical_edges, **opts_edges)
        ax[3].set_title('Vertical Gradients')
        fig.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)

        im5 = ax[4].imshow(diagonal_dots, **opts_matrix)
        ax[4].set_title('Diagonal Squares')
        fig.colorbar(im5, ax=ax[4], fraction=0.046, pad=0.04)

        im6 = ax[5].imshow(hic_matrix, **opts_matrix)
        ax[5].set_title('Reverted Hi-C Map')
        fig.colorbar(im6, ax=ax[5], fraction=0.046, pad=0.04)

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    return diagonal_dots, dots, horizontal_edges, vertical_edges
