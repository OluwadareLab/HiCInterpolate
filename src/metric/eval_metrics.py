
import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from src.metric.genome_disco import compute_reproducibility
from src.metric.genome_disco_gpu import compute_reproducibility_gpu
from src.metric.hicrep import hicrepSCC as hicrep_scc
from src.metric.hicrep_gpu import hicrepSCCGPU as hicrep_scc_gpu
from scipy.sparse import csr_matrix
import numpy as np

_EPSILON = 1e-8
_METRIC_CACHE = {}


def _device_key(device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def _cached_metric(name: str, device, factory):
    key = (name, _device_key(device))
    if key not in _METRIC_CACHE:
        _METRIC_CACHE[key] = factory().to(device)
    return _METRIC_CACHE[key]


def _get_psnr_module(device, data_range: float = 1.0):
    return _cached_metric(
        ("psnr", data_range),
        device,
        lambda: PeakSignalNoiseRatio(data_range=data_range),
    )


def _get_ssim_module(device, data_range: float = 1.0):
    return _cached_metric(
        ("ssim", data_range),
        device,
        lambda: StructuralSimilarityIndexMeasure(data_range=data_range),
    )


def _get_lpips_module(device):
    return _cached_metric(
        "lpips",
        device,
        lambda: LearnedPerceptualImagePatchSimilarity(net_type="vgg"),
    )


@torch.no_grad()
def get_psnr(preds: Tensor, target: Tensor, data_range: float = 1.0):
    psnr = _get_psnr_module(preds.device, data_range)
    return psnr(preds, target)


@torch.no_grad()
def get_ssim(preds: Tensor, target: Tensor, data_range: float = 1.0):
    ssim = _get_ssim_module(preds.device, data_range)
    return ssim(preds, target)


@torch.no_grad()
def get_genome_disco(preds: Tensor, target: Tensor):
    if preds.device != torch.device("cpu"):
        return compute_reproducibility_gpu(preds, target, True)
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
def get_hicrep(preds: Tensor, target: Tensor):
    if preds.device != torch.device("cpu"):
        return hicrep_scc_gpu(target, preds)
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
def get_lpips(preds, target):
    lpips = _get_lpips_module(preds.device)
    preds_min = preds.amin(dim=(1, 2, 3), keepdim=True)
    preds_max = preds.amax(dim=(1, 2, 3), keepdim=True)
    preds_norm = (preds - preds_min) / (preds_max - preds_min + _EPSILON)

    tmp_preds = preds_norm.repeat(1, 3, 1, 1)
    tmp_target = target.repeat(1, 3, 1, 1)
    return lpips(tmp_preds, tmp_target)


GPU_METRIC_FUNCS = {
    "psnr": get_psnr,
    "ssim": get_ssim,
    "genome_disco": get_genome_disco,
    "hicrep": get_hicrep,
    "lpips": get_lpips,
}

# Aliases used by test_lib / train_lib
get_psnr_gpu = get_psnr
get_ssim_gpu = get_ssim
get_genome_disco_gpu = get_genome_disco
get_hicrep_gpu = get_hicrep
get_lpips_gpu = get_lpips


@torch.no_grad()
def get_metric_gpu(metric_name: str, preds: Tensor, target: Tensor):
    if metric_name not in GPU_METRIC_FUNCS:
        valid_metrics = ", ".join(GPU_METRIC_FUNCS)
        raise ValueError(
            f"Unknown GPU metric '{metric_name}'. Valid metrics: {valid_metrics}")
    return GPU_METRIC_FUNCS[metric_name](preds, target)


@torch.no_grad()
def get_eval_metrics_gpu(preds: Tensor, target: Tensor, include_lpips: bool = True):
    metrics = {
        "psnr": get_psnr(preds, target),
        "ssim": get_ssim(preds, target),
        "genome_disco": get_genome_disco(preds, target),
        "hicrep": get_hicrep(preds, target),
    }
    if include_lpips:
        metrics["lpips"] = get_lpips(preds, target)
    return metrics
