
import torch
from torch import Tensor
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from src.metric.genome_disco import compute_reproducibility
from src.metric.genome_disco_gpu import compute_reproducibility_gpu
from src.metric.hicrep import hicrepSCC as hicrep_scc
from src.metric.hicrep_gpu import hicrepSCCGPU as hicrep_scc_gpu
from scipy.sparse import csr_matrix
import numpy as np

_EPSILON = 1e-8
_LPIPS_CACHE = {}


def get_psnr(preds: Tensor, target: Tensor, data_range: float = 1.0):
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(preds.device)
    psnr_score = psnr(preds, target)
    return psnr_score


@torch.no_grad()
def get_psnr_gpu(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_psnr(preds, target, data_range=data_range)


def get_ssim(preds: Tensor, target: Tensor, data_range: float = 1.0):
    ssim = StructuralSimilarityIndexMeasure(
        data_range=data_range).to(preds.device)
    ssim_score = ssim(preds, target)
    return ssim_score


@torch.no_grad()
def get_ssim_gpu(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_ssim(preds, target, data_range=data_range)


def get_ms_ssim(preds: Tensor, target: Tensor, data_range: float = 1.0):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=data_range,
        betas=(0.0448, 0.2856, 0.3001),
    ).to(preds.device)
    return ms_ssim(preds, target)


@torch.no_grad()
def get_ms_ssim_gpu(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_ms_ssim(preds, target, data_range=data_range)


get_msssim = get_ms_ssim
get_msssim_gpu = get_ms_ssim_gpu


def get_genome_disco(preds: Tensor, target: Tensor):
    repro_list = []
    for p, t in zip(preds, target):
        p_np = p.squeeze(0).detach().cpu().numpy()
        p_csr = csr_matrix(p_np)
        y_np = t.squeeze(0).detach().cpu().numpy()
        y_csr = csr_matrix(y_np)
        repro = compute_reproducibility(p_csr, y_csr, True)
        repro_list.append(repro)
    genome_disco_score = np.mean(repro_list)
    genome_disco_score = torch.tensor(
        genome_disco_score).float().to(preds.device)
    return genome_disco_score


@torch.no_grad()
def get_genome_disco_gpu(preds: Tensor, target: Tensor):
    return compute_reproducibility_gpu(preds, target, True)


def get_hicrep(preds: Tensor, target: Tensor):
    scc_list = []
    for p, t in zip(preds, target):
        p_np = p.squeeze(0).detach().cpu().numpy()
        y_np = t.squeeze(0).detach().cpu().numpy()
        scc = hicrep_scc(y_np, p_np)
        scc_list.append(scc)
    hicrep_score = np.mean(scc_list)
    hicrep_score = torch.tensor(
        hicrep_score).float().to(preds.device)
    return hicrep_score


@torch.no_grad()
def get_hicrep_gpu(preds: Tensor, target: Tensor):
    return hicrep_scc_gpu(target, preds)


def get_ent3c(preds: Tensor, target: Tensor):
    ent3c_score = torch.tensor(
        0.0).float().to(preds.device)
    return ent3c_score


def _minmax_01(x: Tensor):
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + _EPSILON)


def _get_lpips_module(device: torch.device):
    key = str(device)
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
            normalize=True,
        ).to(device).eval()
    return _LPIPS_CACHE[key]


def get_lpips(preds: Tensor, target: Tensor):
    lpips = _get_lpips_module(preds.device)
    tmp_preds = _minmax_01(preds).repeat(1, 3, 1, 1)
    tmp_target = _minmax_01(target).repeat(1, 3, 1, 1)
    lpips_score = lpips(tmp_preds, tmp_target)
    return lpips_score


@torch.no_grad()
def get_lpips_gpu(preds: Tensor, target: Tensor):
    return get_lpips(preds, target)


@torch.no_grad()
def get_sparse_support_metrics(preds: Tensor, target: Tensor, threshold: float = 1e-3):
    pred_support = preds > threshold
    target_support = target > 0

    tp = (pred_support & target_support).sum().float()
    fp = (pred_support & ~target_support).sum().float()
    fn = (~pred_support & target_support).sum().float()

    precision = tp / (tp + fp + _EPSILON)
    recall = tp / (tp + fn + _EPSILON)
    f1 = 2.0 * precision * recall / (precision + recall + _EPSILON)
    pred_density = pred_support.float().mean()
    target_density = target_support.float().mean()
    density_error = torch.abs(pred_density - target_density)

    nonzero = target_support.float()
    zero = (~target_support).float()
    nonzero_mae = (torch.abs(preds - target) * nonzero).sum() / nonzero.sum().clamp_min(1.0)
    zero_mae = (torch.abs(preds) * zero).sum() / zero.sum().clamp_min(1.0)

    return {
        "sparse_precision": precision,
        "sparse_recall": recall,
        "sparse_f1": f1,
        "pred_density": pred_density,
        "target_density": target_density,
        "density_error": density_error,
        "nonzero_mae": nonzero_mae,
        "zero_mae": zero_mae,
    }


def _rank_average_ties(x: Tensor) -> Tensor:
    ranks = torch.empty_like(x, dtype=torch.float32)
    positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32)

    for i in range(x.size(0)):
        values, order = torch.sort(x[i])
        _, counts = torch.unique_consecutive(values, return_counts=True)
        starts = torch.cumsum(
            torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
        ends = starts + counts - 1
        avg_ranks = (starts.to(torch.float32) + ends.to(torch.float32)) / 2.0
        sorted_ranks = torch.repeat_interleave(avg_ranks, counts)
        sample_ranks = torch.empty_like(positions)
        sample_ranks.scatter_(0, order, sorted_ranks)
        ranks[i] = sample_ranks

    return ranks


def get_scc(pred: Tensor, target: Tensor, eps=1e-8):
    assert pred.shape == target.shape, "Input shapes must match"
    B, C, H, W = pred.shape
    assert H == W, "Matrix must be square"

    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    pred_rank = _rank_average_ties(pred_flat)
    target_rank = _rank_average_ties(target_flat)
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


@torch.no_grad()
def get_scc_gpu(preds: Tensor, target: Tensor):
    return get_scc(preds, target)


GPU_METRIC_FUNCS = {
    "psnr": get_psnr_gpu,
    "ssim": get_ssim_gpu,
    "ms_ssim": get_ms_ssim_gpu,
    "msssim": get_ms_ssim_gpu,
    "genome_disco": get_genome_disco_gpu,
    "hicrep": get_hicrep_gpu,
    "scc": get_scc_gpu,
    "lpips": get_lpips_gpu,
}


@torch.no_grad()
def get_metric_gpu(metric_name: str, preds: Tensor, target: Tensor):
    if metric_name not in GPU_METRIC_FUNCS:
        valid_metrics = ", ".join(GPU_METRIC_FUNCS)
        raise ValueError(f"Unknown GPU metric '{metric_name}'. Valid metrics: {valid_metrics}")
    return GPU_METRIC_FUNCS[metric_name](preds, target)


@torch.no_grad()
def get_eval_metrics_gpu(preds: Tensor, target: Tensor, include_lpips: bool = True):
    metrics = {
        "psnr": get_psnr_gpu(preds, target),
        "ssim": get_ssim_gpu(preds, target),
        "ms_ssim": get_ms_ssim_gpu(preds, target),
        "genome_disco": get_genome_disco_gpu(preds, target),
        "hicrep": get_hicrep_gpu(preds, target),
        "scc": get_scc_gpu(preds, target),
    }
    if include_lpips:
        metrics["lpips"] = get_lpips_gpu(preds, target)
    return metrics


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
