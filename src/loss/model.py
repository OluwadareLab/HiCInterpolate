import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchvision.models import vgg19, VGG19_Weights
from src.metric import metrics

def vgg19_features():
    if VGG19_Weights is None:
        return vgg19(pretrained=True).features
    try:
        return vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    except TypeError:
        return vgg19(pretrained=True).features


class CharbonnierLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, y: Tensor, epsilon=1e-3):
        diff = pred - y
        loss = torch.mean(torch.sqrt(diff ** 2 + epsilon ** 2))
        return loss


class AdaptiveWingLoss2D(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1,
                 reduction="mean"):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, reduction: str = None):
        reduction = self.reduction if reduction is None else reduction
        delta = (pred - target).abs()
        target_safe = target.clamp(min=0.0, max=self.alpha - 1e-3)
        exponent = self.alpha - target_safe

        theta = torch.as_tensor(
            self.theta, device=pred.device, dtype=pred.dtype)
        eps = torch.as_tensor(
            self.epsilon, device=pred.device, dtype=pred.dtype)
        theta_eps = theta / eps

        small = delta < theta
        small_loss = self.omega * torch.log1p((delta / eps).pow(exponent))

        a = (
            self.omega
            * exponent
            * theta_eps.pow(exponent - 1.0)
            / (eps * (1.0 + theta_eps.pow(exponent)))
        )
        c = theta * a - self.omega * torch.log1p(theta_eps.pow(exponent))
        large_loss = a * delta - c
        loss = torch.where(small, small_loss, large_loss)

        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss.mean()


class DistanceWeightedLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, scale=0.05, max_weight=10.0):
        super().__init__()
        self.base_loss = base_loss
        self.scale = scale
        self.max_weight = max_weight

    def forward(self, pred: Tensor, target: Tensor):
        raw_loss = self.base_loss(pred, target, reduction="none")
        h, w = target.shape[-2:]
        rows = torch.arange(h, device=target.device).view(h, 1)
        cols = torch.arange(w, device=target.device).view(1, w)
        distance = (rows - cols).abs().to(dtype=target.dtype)
        weights = torch.exp(distance * self.scale).clamp(max=self.max_weight)
        weights = weights / weights.mean().clamp_min(1e-6)
        return (raw_loss * weights.view(1, 1, h, w)).mean()


class StratifiedGenomicLossWrapper(nn.Module):
    def __init__(self, base_loss: nn.Module, eps=1e-6, prior=0.05,
                 max_weight=8.0):
        super().__init__()
        self.base_loss = base_loss
        self.eps = eps
        self.prior = prior
        self.max_weight = max_weight

    def forward(self, pred: Tensor, target: Tensor):
        raw_loss = self.base_loss(pred, target, reduction="none")
        h, w = target.shape[-2:]
        rows = torch.arange(h, device=target.device).view(h, 1)
        cols = torch.arange(w, device=target.device).view(1, w)
        distance = (rows - cols).abs().long()

        flat_dist = distance.reshape(-1)
        flat_active = (target > self.eps).float().mean(dim=(0, 1)).reshape(-1)
        counts = torch.bincount(flat_dist, minlength=max(h, w)).clamp_min(1)
        occupancy = torch.bincount(
            flat_dist, weights=flat_active, minlength=max(h, w)
        ) / counts

        diag_weights = 1.0 / (occupancy + self.prior)
        weights = diag_weights[distance].to(dtype=target.dtype)
        weights = weights / weights.mean().clamp_min(self.eps)
        weights = weights.clamp(max=self.max_weight).view(1, 1, h, w)
        return (raw_loss * weights).mean()


class SymmetryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor):
        transposed = pred.transpose(-1, -2)
        diff = pred - transposed
        loss = torch.abs(diff).mean()
        return loss


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x, target=None):
        if target is None:
            target = x.detach()

        x_sym = 0.5 * (x + x.transpose(-2, -1))
        target_sym = 0.5 * (target + target.transpose(-2, -1))

        diff_h = x_sym[:, :, 1:, :] - x_sym[:, :, :-1, :]
        diff_w = x_sym[:, :, :, 1:] - x_sym[:, :, :, :-1]
        target_h = target_sym[:, :, 1:, :] - target_sym[:, :, :-1, :]
        target_w = target_sym[:, :, :, 1:] - target_sym[:, :, :, :-1]

        h_loss = F.l1_loss(diff_h, target_h)
        w_loss = F.l1_loss(diff_w, target_w)

        return 0.5 * (h_loss + w_loss)


class StyleLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        vgg = vgg19_features()
        vgg.eval()

        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:18])
        self.slice4 = nn.Sequential(*vgg[18:27])
        self.slice5 = nn.Sequential(*vgg[27:36])

        self.normalize = MeanShift(
            data_mean=[0.485, 0.456, 0.406],
            data_std=[0.229, 0.224, 0.225],
            data_range=1.0,
            norm=True,
        )

        self.criterion = nn.MSELoss()
        self.weights = weights if weights is not None else [
            1.0, 1.0, 1.0, 1.0, 1.0]

        for param in self.parameters():
            param.requires_grad = False

    def gram_matrix(self, features: Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        f = features.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, pred: Tensor, target: Tensor):
        X = self.normalize(pred)
        Y = self.normalize(target)

        loss = pred.new_zeros(())
        for weight, slice_layer in zip(
            self.weights,
            (self.slice1, self.slice2, self.slice3, self.slice4, self.slice5),
        ):
            X = slice_layer(X)
            with torch.no_grad():
                Y = slice_layer(Y)
            loss = loss + weight * \
                self.criterion(self.gram_matrix(X), self.gram_matrix(Y))

        return loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1.0, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)

        std = torch.Tensor(data_std)
        mean = torch.Tensor(data_mean)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)

        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1.0 * data_range * mean
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * mean

        for param in self.parameters():
            param.requires_grad = False


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()

        vgg = vgg19_features()
        vgg.eval()

        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:18])
        self.slice4 = nn.Sequential(*vgg[18:27])
        self.slice5 = nn.Sequential(*vgg[27:36])

        self.normalize = MeanShift(
            data_mean=[0.485, 0.456, 0.406],
            data_std=[0.229, 0.224, 0.225],
            data_range=1.0,
            norm=True,
        )

        self.register_buffer(
            "layer_weights",
            torch.tensor([1.0 / 2.6, 1.0 / 4.8, 1.0 /
                         3.7, 1.0 / 5.6, 1.0 / 1.5]),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        X = self.normalize(pred)
        with torch.no_grad():
            Y = self.normalize(target)

        loss = pred.new_zeros(())
        for weight, slice_layer in zip(
            self.layer_weights,
            (self.slice1, self.slice2, self.slice3, self.slice4, self.slice5),
        ):
            X = slice_layer(X)
            with torch.no_grad():
                Y = slice_layer(Y)
            loss = loss + weight * F.l1_loss(X, Y)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.min() < 0:
            pred = torch.sigmoid(pred)
        pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)

        target = target.clamp(0.0, 1.0)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_coeff


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor):
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    



class CombinedLoss(nn.Module):
    _triu_cache: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        loss_names = {weight_params["name"]
                      for weight_params in cfg.loss.weight_parameters}
        self.vgg_loss = VGGPerceptualLoss().to(
            cfg.device) if "vgg" in loss_names else None
        self.style_loss = StyleLoss().to(cfg.device) if "style" in loss_names else None
        self.l1_loss = nn.L1Loss().to(cfg.device)
        self.mse_loss = nn.MSELoss().to(cfg.device)
        self.tv_loss = TVLoss().to(cfg.device)
        self.symmetry_loss = SymmetryLoss().to(cfg.device)
        ssim_cls = StructuralSimilarityIndexMeasure
        ms_ssim_cls = MultiScaleStructuralSimilarityIndexMeasure
        self.ssim = ssim_cls(data_range=1.0).to(cfg.device)
        self.ms_ssim = ms_ssim_cls(
            data_range=1.0,
            kernel_size=3,
            betas=(0.0448, 0.2856, 0.3001, 0.3695, 0.4302),
            reduction="elementwise_mean"
        ).to(cfg.device)
        self.aw_loss = AdaptiveWingLoss2D().to(cfg.device)
        self.dw_loss = DistanceWeightedLoss(self.aw_loss).to(cfg.device)
        self.stratified_loss = StratifiedGenomicLossWrapper(
            self.aw_loss).to(cfg.device)
        self.focal_loss = FocalLoss().to(cfg.device)
        self.dice_loss = DiceLoss().to(cfg.device)

    @staticmethod
    def _to_3ch(x: Tensor) -> Tensor:
        return x if x.shape[1] == 3 else x.expand(-1, 3, -1, -1).contiguous()

    @staticmethod
    def weight_schedule(weight_params: dict, epoch: int) -> float:
        for i, boundary in enumerate(weight_params["boundaries"]):
            if epoch < boundary:
                return weight_params["values"][i]
        return weight_params["values"][-1]

    @classmethod
    def dense_sparse_loss(cls, pred, target):
        zero_mask = (target <= 0).float()
        nonzero_mask = (target > 0).float()
        dense_loss = (pred - target).abs().mul(nonzero_mask).sum()
        dense_loss = dense_loss / nonzero_mask.sum().clamp_min(1.0)
        sparse_loss = pred.square().mul(zero_mask).sum()
        sparse_loss = sparse_loss / zero_mask.sum().clamp_min(1.0)
        return dense_loss + 2.0 * sparse_loss

    @classmethod
    def dwpc_loss(cls, pred, target):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        if pred.dim() != 3 or pred.shape != target.shape:
            raise ValueError(
                "DWPC loss expects matching [B, N, N] or [B, 1, N, N] tensors")

        n = pred.shape[-1]
        cache_key = (n, pred.device)
        if cache_key not in cls._triu_cache:
            cls._triu_cache[cache_key] = torch.triu_indices(
                n, n, offset=1, device=pred.device)
        rows, cols = cls._triu_cache[cache_key]
        weights = (cols - rows).to(dtype=pred.dtype).view(1, -1)

        x = pred[:, rows, cols]
        y = target[:, rows, cols]
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mx = (weights * x).sum(dim=1, keepdim=True) / weight_sum
        my = (weights * y).sum(dim=1, keepdim=True) / weight_sum
        cov = (weights * (x - mx) * (y - my)).sum(dim=1)
        var_x = (weights * (x - mx).square()).sum(dim=1)
        var_y = (weights * (y - my).square()).sum(dim=1)
        corr = cov / torch.sqrt(var_x * var_y + 1e-8)
        return corr.clamp(-1.0, 1.0).mean()

    @staticmethod
    def bce_loss(pred_mask: Tensor, gt_mask: Tensor):
        if pred_mask.shape != gt_mask.shape:
            gt_mask = gt_mask.expand_as(pred_mask)
        if pred_mask.detach().amin() >= 0.0 and pred_mask.detach().amax() <= 1.0:
            return F.binary_cross_entropy(pred_mask.clamp(1e-8, 1.0 - 1e-8),
                                          gt_mask)
        return F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    

    def forward(self, pred: Tensor, y: Tensor, epoch: int,
                pred_mask: Tensor = None, gt_mask: Tensor = None):
        loss = pred.new_zeros(())

        for weight_params in self.cfg.loss.weight_parameters:
            weight = self.weight_schedule(
                weight_params=weight_params, epoch=epoch)

            if weight <= 0.0:
                continue
            if weight_params["name"] == "l1":
                loss = loss + weight * self.l1_loss(pred, y)
            elif weight_params["name"] == "l1_reg":
                loss = loss + weight * torch.mean(torch.abs(pred))
            elif weight_params["name"] == "mse":
                log_pred = torch.log1p(pred)
                log_tgt = torch.log1p(y)
                w = (y > 0).float() * 9 + 1
                mse_loss = (w * (log_pred - log_tgt) ** 2).mean()
                loss = loss + weight * mse_loss
            elif weight_params["name"] == "ssim":
                loss = loss + weight * (1.0 - self.ssim(pred, y))
            elif weight_params["name"] == "vgg":
                loss = loss + weight * \
                    self.vgg_loss(self._to_3ch(pred), self._to_3ch(y))
            elif weight_params["name"] == "style":
                loss = loss + weight * \
                    self.style_loss(self._to_3ch(pred), self._to_3ch(y))
            elif weight_params["name"] == "tv":
                loss = loss + weight * self.tv_loss(pred, y)
            elif weight_params["name"] == "symmetry":
                loss = loss + weight * self.symmetry_loss(pred)
            elif weight_params["name"] == "ms_ssim":
                loss = loss + weight * (1.0 - self.ms_ssim(pred, y))
            elif weight_params["name"] == "dwpc":
                loss = loss + weight * (1.0 - self.dwpc_loss(pred, y))
            elif weight_params["name"] == "ds":
                loss = loss + weight * self.dense_sparse_loss(pred, y)
            elif weight_params["name"] == "awl":
                loss = loss + weight * self.aw_loss(pred, y)
            elif weight_params["name"] == "distance_awl":
                loss = loss + weight * self.dw_loss(pred, y)
            elif weight_params["name"] == "stratified":
                loss = loss + weight * self.stratified_loss(pred, y)
            elif weight_params["name"] == "bce":
                if pred_mask is None or gt_mask is None:
                    raise ValueError(
                        "bce loss requires pred_mask and gt_mask arguments")
                loss = loss + weight * self.bce_loss(pred_mask, gt_mask)
            elif weight_params["name"] == "focal":
                if pred_mask is not None and gt_mask is not None:
                    loss = loss + weight * self.focal_loss(pred_mask, gt_mask)
                else:
                    loss = loss + weight * self.focal_loss(pred, y)
            elif weight_params["name"] == "dice":
                if pred_mask is None or gt_mask is None:
                    raise ValueError(
                        "dice loss requires pred_mask and gt_mask arguments")
                loss = loss + weight * self.dice_loss(pred_mask, gt_mask)
            elif weight_params["name"] == "lpips":
                loss = loss + weight * metrics.get_lpips_gpu(pred, y)
            else:
                raise ValueError(f"Invalid loss name: {weight_params['name']}")

        return loss
