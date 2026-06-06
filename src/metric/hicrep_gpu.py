import torch
import torch.nn.functional as F
from torch import Tensor


def _as_batched_square_matrix(mat: Tensor) -> Tensor:
    if mat.dim() == 2:
        mat = mat.unsqueeze(0)
    elif mat.dim() == 4 and mat.size(1) == 1:
        mat = mat.squeeze(1)

    if mat.dim() != 3:
        raise ValueError(
            "HiCRep input must have shape [N, N], [B, N, N], or [B, 1, N, N]"
        )
    if mat.size(-1) != mat.size(-2):
        raise ValueError("HiCRep input matrices must be square")
    return mat


def _var_vstran(n: Tensor) -> Tensor:
    """GPU equivalent of hicrep.varVstran."""
    return torch.where(
        n < 2,
        torch.full_like(n, float("nan")),
        (1.0 + 1.0 / n) / 12.0,
    )


def _trim_diags(mat: Tensor, i_diag_max: int, keep_main: bool = False) -> Tensor:
    """GPU equivalent of hicrep.trimDiags for dense tensors."""
    n = mat.size(-1)
    idx = torch.arange(n, device=mat.device)
    g_dist = (idx[:, None] - idx[None, :]).abs()
    keep = (g_dist < i_diag_max) & (keep_main | (g_dist != 0))
    return mat * keep.to(mat.dtype)


def _mean_filter_dense(mat: Tensor, h: int) -> Tensor:
    """GPU equivalent of hicrep.meanFilterSparse."""
    if h <= 0:
        return mat

    kernel_size = 2 * h + 1
    kernel = mat.new_ones((1, 1, kernel_size, kernel_size))
    filtered = F.conv2d(mat.unsqueeze(1), kernel, padding=h).squeeze(1)

    n = mat.size(-1)
    idx = torch.arange(n, device=mat.device)
    row_dist_to_edge = torch.minimum(idx, n - 1 - idx)
    n_dim1 = h + 1 + torch.minimum(row_dist_to_edge, idx.new_full((n,), h))
    col_dist_to_edge = row_dist_to_edge
    n_dim2 = h + 1 + torch.minimum(col_dist_to_edge, idx.new_full((n,), h))
    n_neighbors = (n_dim1[:, None] * n_dim2[None, :]).to(mat.dtype)
    return filtered / n_neighbors


def _scc_by_diag_dense(m1: Tensor, m2: Tensor, n_diags: int) -> Tensor:
    """GPU equivalent of hicrep.sccByDiag + hicrep.upperDiagCsr."""
    batch_size = m1.size(0)
    numerator = m1.new_zeros(batch_size)
    denominator = m1.new_zeros(batch_size)

    for diag in range(1, n_diags):
        x = m1.diagonal(offset=diag, dim1=-2, dim2=-1)
        y = m2.diagonal(offset=diag, dim1=-2, dim2=-1)

        mask = (x + y) != 0
        x = torch.where(mask, x, torch.zeros_like(x))
        y = torch.where(mask, y, torch.zeros_like(y))

        n_samples = mask.sum(dim=1).to(m1.dtype)
        row_sum_m1 = x.sum(dim=1)
        row_sum_m2 = y.sum(dim=1)

        cov = (x * y).sum(dim=1) - row_sum_m1 * row_sum_m2 / n_samples
        var_m1 = x.square().sum(dim=1) - row_sum_m1.square() / n_samples
        var_m2 = y.square().sum(dim=1) - row_sum_m2.square() / n_samples
        rho = cov / torch.sqrt(var_m1 * var_m2)
        weights = n_samples * _var_vstran(n_samples)

        rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        numerator = numerator + rho * weights
        denominator = denominator + weights

    return numerator / denominator


@torch.no_grad()
def hicrepSCCGPU(mat1: Tensor, mat2: Tensor, h: int = 0) -> Tensor:
    """Torch/GPU equivalent of hicrep.hicrepSCC for dense Hi-C tensors.

    Mirrors the CPU pipeline: trim diagonals, optional mean filter, then
    weighted upper-diagonal stratum-corrected correlations.
    """
    out_dtype = mat1.dtype if torch.is_floating_point(mat1) else torch.float32
    device = mat1.device

    mat1 = _as_batched_square_matrix(mat1)
    mat2 = _as_batched_square_matrix(mat2)
    if mat1.shape != mat2.shape:
        raise ValueError("HiCRep input shapes must match")

    mat1 = mat1.to(dtype=torch.float64)
    mat2 = mat2.to(dtype=torch.float64)

    n_diags = mat1.size(-1)
    mat1 = _trim_diags(mat1, n_diags, keep_main=False)
    mat2 = _trim_diags(mat2, n_diags, keep_main=False)

    if h > 0:
        mat1 = _mean_filter_dense(mat1, h)
        mat2 = _mean_filter_dense(mat2, h)

    per_sample = _scc_by_diag_dense(mat1, mat2, n_diags)
    return per_sample.mean().to(device=device, dtype=out_dtype)


hicrep_scc_gpu = hicrepSCCGPU
