import torch
import torch.nn.functional as F
from torch import Tensor


def _as_batched_square_matrix(mat: Tensor) -> Tensor:
    if mat.dim() == 2:
        mat = mat.unsqueeze(0)
    elif mat.dim() == 4 and mat.size(1) == 1:
        mat = mat.squeeze(1)

    if mat.dim() != 3:
        raise ValueError("HiCRep input must have shape [N, N], [B, N, N], or [B, 1, N, N]")
    if mat.size(-1) != mat.size(-2):
        raise ValueError("HiCRep input matrices must be square")
    return mat


def _mean_filter_dense(mat: Tensor, h: int) -> Tensor:
    if h <= 0:
        return mat

    kernel_size = 2 * h + 1
    kernel = mat.new_ones((1, 1, kernel_size, kernel_size))
    filtered = F.conv2d(mat.unsqueeze(1), kernel, padding=h).squeeze(1)

    n = mat.size(-1)
    idx = torch.arange(n, device=mat.device)
    dist_to_edge = torch.minimum(idx, n - 1 - idx)
    n_dim = h + 1 + torch.minimum(dist_to_edge, idx.new_full((n,), h))
    n_neighbors = (n_dim[:, None] * n_dim[None, :]).to(mat.dtype)
    return filtered / n_neighbors


def _trim_main_diag_dense(mat: Tensor) -> Tensor:
    n = mat.size(-1)
    keep = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat * keep.to(mat.dtype)


def _stack_padded_upper_diagonals(mat: Tensor) -> Tensor:
    """Stack upper off-diagonals (offsets 1..N-1) padded to width N-1."""
    n = mat.size(-1)
    if n <= 1:
        return mat.new_zeros(mat.size(0), 0, 0)

    batch_size = mat.size(0)
    width = n - 1
    stacked = mat.new_zeros(batch_size, n - 1, width)
    for offset in range(1, n):
        diag = mat.diagonal(offset=offset, dim1=-2, dim2=-1)
        stacked[:, offset - 1, : diag.size(-1)] = diag
    return stacked


def _scc_by_diag_dense(m1: Tensor, m2: Tensor) -> Tensor:
    x = _stack_padded_upper_diagonals(m1)
    y = _stack_padded_upper_diagonals(m2)
    if x.numel() == 0:
        return m1.new_zeros(m1.size(0))

    mask = (x + y) != 0
    x = torch.where(mask, x, torch.zeros_like(x))
    y = torch.where(mask, y, torch.zeros_like(y))

    n_samples = mask.sum(dim=2).to(m1.dtype)
    row_sum_m1 = x.sum(dim=2)
    row_sum_m2 = y.sum(dim=2)

    cov = (x * y).sum(dim=2) - row_sum_m1 * row_sum_m2 / n_samples
    var_m1 = x.square().sum(dim=2) - row_sum_m1.square() / n_samples
    var_m2 = y.square().sum(dim=2) - row_sum_m2.square() / n_samples
    rho = cov / torch.sqrt(var_m1 * var_m2)

    weights = n_samples * torch.where(
        n_samples < 2,
        torch.full_like(n_samples, float("nan")),
        (1.0 + 1.0 / n_samples) / 12.0,
    )

    rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    numerator = (rho * weights).sum(dim=1)
    denominator = weights.sum(dim=1)
    scc = torch.zeros_like(numerator)
    valid = denominator > 0
    scc[valid] = numerator[valid] / denominator[valid]
    return scc


@torch.no_grad()
def hicrepSCCGPU(mat1: Tensor, mat2: Tensor, h: int = 0) -> Tensor:
    """Torch/GPU equivalent of hicrep.hicrepSCC for dense Hi-C tensors."""
    mat1 = _as_batched_square_matrix(mat1)
    mat2 = _as_batched_square_matrix(mat2)
    if mat1.shape != mat2.shape:
        raise ValueError("HiCRep input shapes must match")

    mat1 = _trim_main_diag_dense(mat1)
    mat2 = _trim_main_diag_dense(mat2)
    mat1 = _mean_filter_dense(mat1, h)
    mat2 = _mean_filter_dense(mat2, h)
    return _scc_by_diag_dense(mat1, mat2).nanmean()


hicrep_scc_gpu = hicrepSCCGPU
