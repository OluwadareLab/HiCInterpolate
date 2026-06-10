
import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from src.metric.genome_disco import compute_reproducibility
from src.metric.genome_disco_gpu import compute_reproducibility_gpu
from src.metric.hicrep import hicrepSCC as hicrep_scc
from src.metric.hicrep_gpu import hicrepSCCGPU as hicrep_scc_gpu
from src.metric.ent3c import get_similarity as ent3c_similarity
from scipy.sparse import csr_matrix
import numpy as np

_EPSILON = 1e-8

_PSNR_CACHE: dict = {}
_SSIM_CACHE: dict = {}
_LPIPS_CACHE: dict = {}


def _get_psnr_module(device, data_range: float):
    key = (str(device), data_range)
    if key not in _PSNR_CACHE:
        _PSNR_CACHE[key] = PeakSignalNoiseRatio(data_range=data_range).to(device)
    return _PSNR_CACHE[key]


def _get_ssim_module(device, data_range: float):
    key = (str(device), data_range)
    if key not in _SSIM_CACHE:
        _SSIM_CACHE[key] = StructuralSimilarityIndexMeasure(
            data_range=data_range).to(device)
    return _SSIM_CACHE[key]


def _get_lpips_module(device):
    key = str(device)
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg').to(device)
    return _LPIPS_CACHE[key]


def get_psnr(preds: Tensor, target: Tensor, data_range: float = 1.0):
    psnr = _get_psnr_module(preds.device, data_range)
    psnr_score = psnr(preds, target)
    return psnr_score


@torch.no_grad()
def get_psnr_gpu(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_psnr(preds, target, data_range=data_range)


def get_ssim(preds: Tensor, target: Tensor, data_range: float = 1.0):
    ssim = _get_ssim_module(preds.device, data_range)
    ssim_score = ssim(preds, target)
    return ssim_score


@torch.no_grad()
def get_ssim_gpu(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_ssim(preds, target, data_range=data_range)



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
    scores = []
    for p, t in zip(preds, target):
        p_np = p.squeeze(0).detach().cpu().numpy()
        t_np = t.squeeze(0).detach().cpu().numpy()
        try:
            q = ent3c_similarity(t_np, p_np)
            if q is not None and np.isfinite(q):
                scores.append(float(q))
        except Exception:
            continue
    ent3c_score = float(np.mean(scores)) if scores else 0.0
    return torch.tensor(ent3c_score).float().to(preds.device)


def get_lpips(preds, target):
    lpips = _get_lpips_module(preds.device)
    preds_min = preds.amin(dim=(1, 2, 3), keepdim=True)
    preds_max = preds.amax(dim=(1, 2, 3), keepdim=True)
    preds_norm = (preds - preds_min) / (preds_max - preds_min + _EPSILON)

    tmp_preds = preds_norm.repeat(1, 3, 1, 1)
    tmp_target = target.repeat(1, 3, 1, 1)
    lpips_score = lpips(tmp_preds, tmp_target)
    return lpips_score


@torch.no_grad()
def get_lpips_gpu(preds: Tensor, target: Tensor):
    return get_lpips(preds, target)


@torch.no_grad()
def get_scc_gpu(preds: Tensor, target: Tensor):
    return get_scc(preds, target)


@torch.no_grad()
def get_pcc_gpu(preds: Tensor, target: Tensor):
    return get_pcc(preds, target)


GPU_METRIC_FUNCS = {
    "psnr": get_psnr_gpu,
    "ssim": get_ssim_gpu,
    "genome_disco": get_genome_disco_gpu,
    "hicrep": get_hicrep_gpu,
    "lpips": get_lpips_gpu,
    "scc": get_scc_gpu,
    "pcc": get_pcc_gpu,
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
        "genome_disco": get_genome_disco_gpu(preds, target),
        "hicrep": get_hicrep_gpu(preds, target),
        "scc": get_scc_gpu(preds, target),
        "pcc": get_pcc_gpu(preds, target),
    }
    if include_lpips:
        metrics["lpips"] = get_lpips_gpu(preds, target)
    return metrics


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
