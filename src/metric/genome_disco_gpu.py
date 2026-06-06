import torch
from torch import Tensor


def _as_batched_square_matrix(mat: Tensor) -> Tensor:
    if mat.dim() == 2:
        mat = mat.unsqueeze(0)
    elif mat.dim() == 4 and mat.size(1) == 1:
        mat = mat.squeeze(1)

    if mat.dim() != 3:
        raise ValueError(
            "GenomeDISCO input must have shape [N, N], [B, N, N], or [B, 1, N, N]"
        )
    if mat.size(-1) != mat.size(-2):
        raise ValueError("GenomeDISCO input matrices must be square")
    return mat


def _to_transition_dense(mat: Tensor) -> Tensor:
    rowsums = mat.sum(dim=-1, keepdim=True)
    rowsums = torch.where(rowsums == 0.0, torch.ones_like(rowsums), rowsums)
    return mat / rowsums


def _auc_unit_spacing(scores: Tensor) -> Tensor:
    if scores.size(0) == 1:
        return scores.squeeze(0)
    return 0.5 * (scores[:-1] + scores[1:]).sum(dim=0) / (scores.size(0) - 1)


@torch.no_grad()
def compute_reproducibility_gpu(
    m1: Tensor,
    m2: Tensor,
    transition: bool,
    tmax: int = 3,
    tmin: int = 3,
) -> Tensor:
    """Torch/GPU equivalent of genome_disco.compute_reproducibility.

    This mirrors the current SciPy implementation: symmetrize each matrix,
    optionally row-normalize to transition matrices, compute random-walk
    differences from tmin..tmax, then return 1 - AUC.
    """
    if tmin < 1 or tmax < tmin:
        raise ValueError("GenomeDISCO requires 1 <= tmin <= tmax")

    out_dtype = m1.dtype if torch.is_floating_point(m1) else torch.float32
    device = m1.device

    m1 = _as_batched_square_matrix(m1)
    m2 = _as_batched_square_matrix(m2)
    if m1.shape != m2.shape:
        raise ValueError("GenomeDISCO input shapes must match")

    m1 = m1.to(dtype=torch.float64)
    m2 = m2.to(dtype=torch.float64)

    m1 = m1 + m1.transpose(-1, -2)
    m2 = m2 + m2.transpose(-1, -2)

    if transition:
        m1 = _to_transition_dense(m1)
        m2 = _to_transition_dense(m2)

    rowsums_1 = m1.sum(dim=-1)
    rowsums_2 = m2.sum(dim=-1)
    nonzero_1 = (rowsums_1 > 0.0).sum(dim=-1).to(m1.dtype)
    nonzero_2 = (rowsums_2 > 0.0).sum(dim=-1).to(m1.dtype)
    nonzero_total = 0.5 * (nonzero_1 + nonzero_2)

    scores = []
    rw1 = None
    rw2 = None
    for t in range(1, tmax + 1):
        if t == 1:
            rw1 = m1.clone()
            rw2 = m2.clone()
        else:
            rw1 = torch.bmm(rw1, m1)
            rw2 = torch.bmm(rw2, m2)

        if t >= tmin:
            diff = (rw1 - rw2).abs().sum(dim=(-1, -2))
            scores.append(diff / nonzero_total)

    scores = torch.stack(scores, dim=0)
    if tmin == tmax:
        auc = scores.squeeze(0)
        auc = torch.where(auc > 2.0, torch.full_like(auc, 2.0), auc)
    else:
        auc = _auc_unit_spacing(scores)

    return (1.0 - auc).mean().to(device=device, dtype=out_dtype)
