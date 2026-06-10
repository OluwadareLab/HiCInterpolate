import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchvision.models import vgg19, VGG19_Weights


class CharbonnierLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, y: Tensor, epsilon=1e-3):
        diff = pred - y
        loss = torch.mean(torch.sqrt(diff ** 2 + epsilon ** 2))
        return loss


class StratifiedGenomicLossWrapper(nn.Module):
    def __init__(self, base_loss_module):
        super(StratifiedGenomicLossWrapper, self).__init__()
        self.base_loss = base_loss_module

    def forward(self, pred, target):
        raw_spatial_loss = self.base_loss(pred, target, reduction="none")
        b, c, h, w = target.shape

        rows = torch.arange(h, device=target.device).view(h, 1).repeat(1, w)
        cols = torch.arange(w, device=target.device).view(1, w).repeat(h, 1)
        distance_matrix = torch.abs(rows - cols).float()

        eps = 1e-6
        stratified_weights = torch.zeros_like(distance_matrix)

        for d in range(h):
            mask = (distance_matrix == d)
            occupancy = (target[:, :, mask] > eps).float().mean()
            stratified_weights[mask] = 1.0 / (occupancy + 0.05)

        stratified_weights = stratified_weights / stratified_weights.mean().clamp_min(eps)
        stratified_weights = stratified_weights.clamp(max=8.0).view(1, 1, h, w)

        balanced_loss = raw_spatial_loss * stratified_weights
        return balanced_loss.mean()


class DistanceWeightedWingLoss(nn.Module):
    def __init__(self, base_wing_loss):
        super(DistanceWeightedWingLoss, self).__init__()
        self.base_wing = base_wing_loss

    def forward(self, pred, target):
        raw_loss = self.base_wing(pred, target, reduction="none")

        # 2. Dynamically construct a distance-from-diagonal weight matrix
        b, c, h, w = target.shape

        # Create a meshgrid of coordinates
        rows = torch.arange(h, device=target.device).view(h, 1).repeat(1, w)
        cols = torch.arange(w, device=target.device).view(1, w).repeat(h, 1)

        # Calculate absolute distance from diagonal for each bin entry
        distance_matrix = torch.abs(rows - cols).float()

        # Apply an inverse power-law weight mask: entries further out get amplified
        # This forces the optimizer to fix long-range loop errors instead of ignoring them
        weight_mask = torch.exp(distance_matrix * 0.05).clamp(max=10.0)
        weight_mask = weight_mask.view(1, 1, h, w)  # Broadcastable shape

        # 3. Apply the weight mask to your spatial loss map
        weighted_loss = raw_loss * weight_mask

        return weighted_loss.mean()


class SymmetryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor):
        transposed = pred.transpose(-1, -2)
        diff = pred - transposed
        loss = torch.abs(diff).mean()
        return loss


class AdaptiveWingLoss2D(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1,
                 reduction="mean"):
        """
        Args:
            omega (float): Controls the maximum gradient scale for small errors.
            theta (float): The threshold switching point between the log and linear zones.
            epsilon (float): Avoids division by zero and shapes the internal curve.
            alpha (float): Used in the (alpha - y) exponent to govern background sensitivity.
            reduction (str): mean, sum, or none. Use none for spatial reweighting.
        """
        super(AdaptiveWingLoss2D, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target, reduction=None):
        """
        Args:
            pred (torch.Tensor): Output from the model prediction head [B, 1, H, W]
            target (torch.Tensor): Ground-truth target Hi-C matrix patch [B, 1, H, W]
        """
        delta_y = torch.abs(pred - target)
        reduction = self.reduction if reduction is None else reduction

        small_error_mask = delta_y < self.theta
        large_error_mask = ~small_error_mask

        loss = torch.zeros_like(delta_y)
        target_safe = target.clamp(min=0.0, max=self.alpha - 1e-3)
        theta_over_eps = torch.as_tensor(
            self.theta / self.epsilon, device=target.device, dtype=target.dtype)

        target_small = target_safe[small_error_mask]
        delta_y_small = delta_y[small_error_mask]
        pow_small = self.alpha - target_small
        loss[small_error_mask] = self.omega * torch.log(
            1.0 + torch.pow(delta_y_small / self.epsilon, pow_small)
        )

        target_large = target_safe[large_error_mask]
        delta_y_large = delta_y[large_error_mask]
        pow_large = self.alpha - target_large
        a = (
            self.omega
            * pow_large
            * torch.pow(theta_over_eps, pow_large - 1.0)
            / (self.epsilon * (1.0 + torch.pow(theta_over_eps, pow_large)))
        )
        c = self.theta * a - self.omega * torch.log(
            1.0 + torch.pow(theta_over_eps, pow_large)
        )
        loss[large_error_mask] = a * delta_y_large - c

        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss.mean()


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

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
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
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0, 1.0, 1.0]

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

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
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


class CombinedLoss(nn.Module):
    _triu_cache: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        loss_names = {weight_params["name"] for weight_params in cfg.loss.weight_parameters}
        self.vgg_loss = VGGPerceptualLoss().to(cfg.device) if "vgg" in loss_names else None
        self.style_loss = StyleLoss().to(cfg.device) if "style" in loss_names else None
        self.l1_loss = nn.L1Loss().to(cfg.device)
        self.mse_loss = nn.MSELoss().to(cfg.device)
        self.tv_loss = TVLoss().to(cfg.device)
        self.symmetry_loss = SymmetryLoss().to(cfg.device)
        self.ssim_loss = StructuralSimilarityIndexMeasure(
            data_range=1.0).to(cfg.device)
        self.ms_ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=3,                             # Tightest possible window for fine loops
            # Full 5 scales supported
            betas=(0.0448, 0.2856, 0.3001, 0.3695, 0.4302),
            reduction="elementwise_mean"
        ).to(cfg.device)
        self.aw_loss = AdaptiveWingLoss2D().to(cfg.device)
        self.stratified_loss = StratifiedGenomicLossWrapper(
            self.aw_loss).to(cfg.device)

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
        zero_mask = (target == 0).float()
        nonzero_mask = (target > 0).float()
        dense_loss = F.l1_loss(pred * nonzero_mask,
                               target * nonzero_mask).to(pred.device)
        sparse_penalty = F.mse_loss(
            pred * zero_mask, torch.zeros_like(pred)).to(pred.device)
        lambda_sparse = 2.0
        total_loss = dense_loss + lambda_sparse * sparse_penalty
        return total_loss

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
        weights = (cols - rows).to(dtype=pred.dtype) + 1
        weights = weights.view(1, -1)

        x = pred[:, rows, cols]
        y = target[:, rows, cols]
        weight_sum = weights.sum(dim=1, keepdim=True)
        mx = (weights * x).sum(dim=1, keepdim=True) / weight_sum
        my = (weights * y).sum(dim=1, keepdim=True) / weight_sum
        cov = (weights * (x - mx) * (y - my)).sum(dim=1)
        var_x = (weights * (x - mx) ** 2).sum(dim=1)
        var_y = (weights * (y - my) ** 2).sum(dim=1)
        denom = torch.sqrt(var_x * var_y + 1e-8)
        corr = (cov / denom).clamp(-1.0, 1.0)
        return corr.mean()

    def forward(self, pred: Tensor, y: Tensor, epoch: int,
                pred_mask: Tensor = None, gt_mask: Tensor = None,
                diffusion_noise_pred: Tensor = None,
                diffusion_noise_target: Tensor = None,
                diffusion_mask: Tensor = None):
        loss = pred.new_zeros(())

        for weight_params in self.cfg.loss.weight_parameters:
            weight = self.weight_schedule(
                weight_params=weight_params, epoch=epoch)

            if weight <= 0.0:
                continue
            if weight_params["name"] == "l1":
                loss = loss + weight * self.l1_loss(pred, y)
            elif weight_params["name"] == "mse":
                loss = loss + weight * self.mse_loss(pred, y)
            elif weight_params["name"] == "ssim":
                loss = loss + weight * (1.0 - self.ssim_loss(pred, y))
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
                loss = loss + weight * (1.0 - self.ms_ssim_loss(pred, y))
            elif weight_params["name"] == "dwpc":
                loss = loss + weight * (1.0 - self.dwpc_loss(pred, y))
            elif weight_params["name"] == "ds":
                loss = loss + weight * self.dense_sparse_loss(pred, y)
            elif weight_params["name"] == "awl":
                loss = loss + weight * self.aw_loss(pred, y)
            elif weight_params["name"] == "stratified":
                loss = loss + weight * self.stratified_loss(pred, y)
            elif weight_params["name"] == "bce":
                if pred_mask is None or gt_mask is None:
                    raise ValueError(
                        "bce loss requires pred_mask and gt_mask arguments")
                loss = loss + weight * \
                    F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
            elif weight_params["name"] == "diffusion":
                if diffusion_noise_pred is None or diffusion_noise_target is None:
                    continue
                if diffusion_mask is None:
                    diffusion_mask = torch.ones_like(diffusion_noise_target)
                denom = diffusion_mask.sum().clamp_min(1.0)
                loss = loss + weight * (
                    ((diffusion_noise_pred - diffusion_noise_target) ** 2)
                    * diffusion_mask
                ).sum() / denom
            else:
                raise ValueError(f"Invalid loss name: {weight_params['name']}")

        return loss
