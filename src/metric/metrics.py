
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity, SpatialCorrelationCoefficient
from src.metric.hicrep import compute_hicrep
from src.metric import genome_disco
from src.metric.genomedisco import compute_genomedisco
from scipy import stats
import numpy as np
import torch

_EPSILON = 1e-8
_LPIPS_CACHE = {}


def get_psnr(preds: Tensor, target: Tensor, data_range: float = 1.0):
    max = np.float32(torch.max(preds.max(), target.max()).item())
    psnr = PeakSignalNoiseRatio(data_range=max).to(preds.device)
    psnr_score = psnr(preds, target)
    return psnr_score


def get_psnr_from_tensor(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_psnr(preds, target, data_range=data_range)


def get_ssim(preds: Tensor, target: Tensor, data_range: float = 1.0):
    max = np.float32(torch.max(preds.max(), target.max()).item())
    ssim = StructuralSimilarityIndexMeasure(
        data_range=max).to(preds.device)
    ssim_score = ssim(preds, target)
    return ssim_score


def get_ssim_from_tensor(preds: Tensor, target: Tensor, data_range: float = 1.0):
    return get_ssim(preds, target, data_range=data_range)


def get_ms_ssim(preds: Tensor, target: Tensor):
    max = np.float32(torch.max(preds.max(), target.max()).item())
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=max,
        betas=(0.0448, 0.2856, 0.3001),
    ).to(preds.device)
    return ms_ssim(preds, target)


def get_ms_ssim_from_tensor(preds: Tensor, target: Tensor):
    return get_ms_ssim(preds, target)


def get_spearman(pred: np.ndarray, target: np.ndarray):
    rho, p_val = stats.spearmanr(pred.flatten(), target.flatten())
    return rho


def get_spearman_from_tensor(pred: Tensor, target: Tensor):
    c = 0
    spearman_list = []
    for b in range(pred.shape[0]):
        pred_mat = pred[b, c, :, :].detach().cpu().numpy().flatten()
        target_mat = target[b, c, :, :].detach().cpu().numpy().flatten()
        rho, p_val = stats.spearmanr(pred_mat, target_mat)
        spearman_list.append(rho)

    return np.nanmean(spearman_list)


def get_scc(preds: Tensor, target: Tensor):
    scc = SpatialCorrelationCoefficient().to(preds.device)
    scc_score = scc(preds, target)
    return scc_score


def get_scc_from_tensor(preds: Tensor, target: Tensor):
    return get_scc(preds, target)


def get_genome_disco(preds: np.ndarray, target: np.ndarray):
    return genome_disco.compute_genomedisco(preds, target)


def get_genome_disco_from_tensor(preds: Tensor, target: Tensor):
    return genome_disco.compute_genomedisco_from_tensor(preds, target)


def get_genome_disco2(preds: np.ndarray, target: np.ndarray, resol=None):
    return compute_genomedisco.compute_reproducibility(preds, target, resol=resol)


def get_genome_disco2_from_tensor(preds: Tensor, target: Tensor, resol=None):
    return compute_genomedisco.compute_reproducibility_from_tensor(preds, target, resol=resol)

def get_hicrep(preds: Tensor, target: Tensor, resol=None, patch_size=None, h=None):
    return compute_hicrep.get_hicrep_scc(preds, target, resol, patch_size, h)


def get_hicrep_from_tensor(preds: Tensor, target: Tensor, resol=None, patch_size=None, h=None):
    return compute_hicrep.get_hicrep_scc_from_tensor(preds, target, resol, patch_size, h)


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


GPU_METRIC_FUNCS = {
    "psnr": get_psnr_from_tensor,
    "ssim": get_ssim_from_tensor,
    "ms-ssim": get_ms_ssim_from_tensor,
    "spearman": get_spearman_from_tensor,
    "scc": get_scc_from_tensor,
    "genome_disco": get_genome_disco_from_tensor,
    "genome_disco2": get_genome_disco2_from_tensor,
    "hicrep": get_hicrep_from_tensor,
    "lpips": get_lpips,
}


@torch.no_grad()
def get_metric_gpu(metric_name: str, preds: Tensor, target: Tensor, resol=None, patch_size=None, h=None):
    if metric_name not in GPU_METRIC_FUNCS:
        valid_metrics = ", ".join(GPU_METRIC_FUNCS)
        raise ValueError(
            f"Unknown GPU metric '{metric_name}'. Valid metrics: {valid_metrics}")
    if metric_name == "genome_disco2":
        return GPU_METRIC_FUNCS[metric_name](preds, target, resol=resol)
    if metric_name == "hicrep":
        return GPU_METRIC_FUNCS[metric_name](preds, target, resol=resol, patch_size=patch_size, h=h)
    return GPU_METRIC_FUNCS[metric_name](preds, target)


@torch.no_grad()
def get_eval_metrics_gpu(preds: Tensor, target: Tensor, include_lpips: bool = True):
    metrics = {
        "psnr": get_psnr_from_tensor(preds, target),
        "ssim": get_ssim_from_tensor(preds, target),
        "ms-ssim": get_ms_ssim_from_tensor(preds, target),
        "spearman": get_spearman_from_tensor(preds, target),
        "scc": get_scc_from_tensor(preds, target),
        "genome_disco": get_genome_disco_from_tensor(preds, target),
        "genome_disco2": get_genome_disco2_from_tensor(preds, target),
        "hicrep": get_hicrep_from_tensor(preds, target),
        "lpips": get_lpips(preds, target) if include_lpips else None
    }
    if include_lpips:
        metrics["lpips"] = get_lpips(preds, target)
    return metrics
