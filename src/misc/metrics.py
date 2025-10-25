from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import torch
import torch.nn.functional as F
from torch import nn
from lpips import LPIPS
from scipy import stats
from src.misc.genome_disco import get_avg_genome_disco
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef, spearman_corrcoef
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_psnr(pred, target):
    return peak_signal_noise_ratio(pred, target, data_range=1.0)


def get_ssim(pred, target):
    return structural_similarity_index_measure(pred, target, data_range=1.0)


def get_pcc(pred, target, eps=1e-8):
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    vx = (pred * pred).mean(dim=1)
    vy = (target * target).mean(dim=1)
    corr = (pred * target).mean(dim=1) / torch.sqrt(vx * vy + eps)
    return corr.mean()


def get_scc(pred, target):
    spearman = SpearmanCorrCoef()
    # return spearman(pred, target)
    return spearman_corrcoef(pred.flatten(), target.flatten())


def get_genome_disco(pred, target):
    return get_avg_genome_disco(pred, target)


def get_ncc(pred, target, eps=1e-8):
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    numerator = ((pred_flat - pred_mean) *
                 (target_flat - target_mean)).sum(dim=1)
    denominator = (
        torch.sqrt(((pred_flat - pred_mean) ** 2).sum(dim=1) + eps)
        * torch.sqrt(((target_flat - target_mean) ** 2).sum(dim=1) + eps)
    )
    return (numerator / denominator).mean()


class LPIPSLoss(nn.Module):
    def __init__(self, net='vgg', device='cuda'):
        super().__init__()
        self.lpips_fn = LPIPS(net=net).to(device)

    @torch.no_grad()
    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.lpips_fn(pred, target).mean()


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])
    # Convert max_val to tensor on the same device
    max_val = torch.tensor(max_val, device=pred.device, dtype=pred.dtype)
    psnr_val = 20 * torch.log10(max_val) - 10 * torch.log10(mse + 1e-8)
    return psnr_val.mean()


def ssim(pred, target, window_size=11, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Gaussian window
    def gaussian_window(window_size, sigma=1.5):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    window = gaussian_window(window_size).to(pred.device)
    window = window.expand(pred.size(1), 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size //
                   2, groups=target.size(1))

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(
        pred * pred, window, padding=window_size // 2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window,
                         padding=window_size // 2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window,
                       padding=window_size // 2, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()
