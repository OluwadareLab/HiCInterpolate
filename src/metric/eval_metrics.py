
import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from src.metric.genome_disco import compute_reproducibility
from scipy.sparse import csr_matrix
import numpy as np

_EPSILON = 1e-8


def get_psnr(preds: Tensor, target: Tensor, data_range: float = 1.0):
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(preds.device)
    psnr_score = psnr(preds, target)
    return psnr_score


def get_ssim(preds: Tensor, target: Tensor, data_range: float = 1.0):
    ssim = StructuralSimilarityIndexMeasure(
        data_range=data_range).to(preds.device)
    ssim_score = ssim(preds, target)
    return ssim_score


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


def get_lpips(preds, target):
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg').to(preds.device)
    preds_min = preds.amin(dim=(1, 2, 3), keepdim=True)
    preds_max = preds.amax(dim=(1, 2, 3), keepdim=True)
    preds_norm = (preds - preds_min) / (preds_max - preds_min + _EPSILON)

    tmp_preds = preds_norm.repeat(1, 3, 1, 1)
    tmp_target = target.repeat(1, 3, 1, 1)
    lpips_score = lpips(tmp_preds, tmp_target)
    return lpips_score


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
