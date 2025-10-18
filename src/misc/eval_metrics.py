
from torch import Tensor
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

_EPSILON = 1e-8


def get_psnr(pred, target):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    return psnr(pred, target)


def get_ssim(pred, target):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return ssim(pred, target)


def get_scc(pred, target, eps=1e-8):
    assert pred.shape == target.shape, "Input shapes must match"
    B, C, H, W = pred.shape
    assert H == W, "Matrix must be square"

    B = pred.size(0)
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)

    pred_rank = pred_flat.argsort(dim=1).argsort(dim=1).float()
    target_rank = target_flat.argsort(dim=1).argsort(dim=1).float()
    pred_mean = pred_rank.mean(dim=1, keepdim=True)
    target_mean = target_rank.mean(dim=1, keepdim=True)

    pred_centered = pred_rank - pred_mean
    target_centered = target_rank - target_mean

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((pred_centered**2).sum(dim=1)
                             * (target_centered**2).sum(dim=1) + eps)

    per_sample_rho = numerator / denominator
    mean_scc = per_sample_rho.mean()
    return mean_scc


def get_pcc(pred, target, eps=1e-8):
    assert pred.shape == target.shape, "Input shapes must match"
    B, C, H, W = pred.shape
    assert H == W, "Matrix must be square"
    pred_flatten = pred.view(pred.size(0), -1)
    target_flatten = target.view(target.size(0), -1)

    pred_mean = pred_flatten.mean(dim=1, keepdim=True)
    target_mean = target_flatten.mean(dim=1, keepdim=True)

    pred_norm = pred_flatten - pred_mean
    target_norm = target_flatten - target_mean

    r_num = (pred_norm * target_norm).sum(dim=1)
    r_den = torch.sqrt((pred_norm**2).sum(dim=1) *
                       (target_norm**2).sum(dim=1) + eps)

    pccs = r_num / (r_den + eps)
    mean_pcc = torch.mean(pccs)

    return mean_pcc


def to_transition_torch(matrix, eps=1e-8):
    row_sum = matrix.sum(dim=-1, keepdim=True)
    row_sum = torch.clamp(row_sum, min=eps)
    return matrix / row_sum


def compute_reproducibility(pred, target, transition=True, tmin=3, tmax=3):
    assert pred.shape == target.shape, "Input shapes must match"
    B, C, H, W = pred.shape
    assert H == W, "Matrix must be square"
    pred_sym = pred + pred.transpose(-1, -2)
    target_sym = target + target.transpose(-1, -2)

    if transition:
        pred_sym = to_transition_torch(pred_sym)
        target_sym = to_transition_torch(target_sym)

    row_sum1 = pred_sym.sum(dim=-1)
    row_sum2 = target_sym.sum(dim=-1)
    nonzero1 = (row_sum1 > 0).float().sum(dim=-1)
    nonzero2 = (row_sum2 > 0).float().sum(dim=-1)
    nonzero_total = 0.5 * (nonzero1 + nonzero2)
    nonzero_total = torch.clamp(nonzero_total, min=1e-8)

    rw1 = pred_sym.clone()
    rw2 = target_sym.clone()

    scores = []

    for t in range(1, tmax + 1):
        if t > 1:
            rw1 = torch.matmul(rw1, pred_sym)
            rw2 = torch.matmul(rw2, target_sym)

        if t >= tmin:
            diff = torch.abs(rw1 - rw2).sum(dim=(-2, -1))
            score_t = diff / nonzero_total
            scores.append(score_t)

    scores = torch.stack(scores, dim=-1)
    if tmin == tmax:
        auc = scores[..., 0]
        auc = torch.clamp(auc, 0, 2)
    else:
        f_i = scores[..., :-1]
        f_ip1 = scores[..., 1:]
        auc = 0.5 * (f_i + f_ip1)
        auc = auc.sum(dim=-1) / (scores.shape[-1] - 1)

    reproducibility = 1.0 - auc
    reproducibility = reproducibility.mean(dim=1)

    return reproducibility


def get_genome_disco(pred: Tensor, target: Tensor):
    reprod_list = compute_reproducibility(pred, target)
    mean_repro = torch.mean(reprod_list)
    return mean_repro


def get_ncc(pred, target, eps=1e-8):
    assert pred.shape == target.shape, "Input shapes must match"
    B, C, H, W = pred.shape
    assert H == W, "Matrix must be square"
    pred_flat = pred.view(B, C, -1)
    target_flat = target.view(B, C, -1)

    mean_pred = pred_flat.mean(dim=-1, keepdim=True)
    mean_target = target_flat.mean(dim=-1, keepdim=True)

    predm = pred_flat - mean_pred
    targetm = target_flat - mean_target
    num = (predm * targetm).sum(dim=-1)
    den_pred = torch.sqrt((predm * predm).sum(dim=-1).clamp(min=eps))
    den_target = torch.sqrt((targetm * targetm).sum(dim=-1).clamp(min=eps))

    denom = den_pred * den_target
    ncc = num / denom
    mean_ncc = ncc.view(-1).mean()
    return mean_ncc


def get_lpips(pred, target):
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex').to(pred.device)

    pred_min = pred.amin(dim=(1, 2, 3), keepdim=True)
    pred_max = pred.amax(dim=(1, 2, 3), keepdim=True)
    pred_norm = (pred - pred_min) / (pred_max - pred_min + _EPSILON)

    temp_pred = pred_norm.repeat(1, 3, 1, 1)
    temp_target = target.repeat(1, 3, 1, 1)

    return lpips(temp_pred, temp_target)
