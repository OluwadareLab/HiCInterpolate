import math

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
        # 1. Compute element-wise unreduced spatial loss matrix
        raw_spatial_loss = self.base_loss(pred, target)
        b, c, h, w = target.shape

        # 2. Build a distance stratification normalization mask
        rows = torch.arange(h, device=target.device).view(h, 1).repeat(1, w)
        cols = torch.arange(w, device=target.device).view(1, w).repeat(h, 1)
        distance_matrix = torch.abs(rows - cols).float()

        # Calculate the empirical mean loss for each distance band in the target
        # To avoid zero division, we add a tiny stability epsilon
        eps = 1e-6
        stratified_weights = torch.zeros_like(distance_matrix)

        for d in range(h):
            mask = (distance_matrix == d)
            target_band_mean = target[:, :, mask].mean()
            # If the target band is nearly empty, we amplify the loss weight
            # to force the model to capture the sparse structure
            stratified_weights[mask] = 1.0 / (target_band_mean + eps)

        # Clamp max weights to avoid gradient explosions far out
        stratified_weights = stratified_weights.clamp(
            max=50.0).view(1, 1, h, w)

        # 3. Apply the stratified mask to the loss landscape
        balanced_loss = raw_spatial_loss * stratified_weights
        return balanced_loss.mean()


class DistanceWeightedWingLoss(nn.Module):
    def __init__(self, base_wing_loss):
        super(DistanceWeightedWingLoss, self).__init__()
        self.base_wing = base_wing_loss

    def forward(self, pred, target):
        # 1. Compute standard pixel-level loss matrix (unreduced)
        # Ensure your AdaptiveWingLoss returns an unreduced tensor [B, 1, H, W]
        raw_loss = self.base_wing(pred, target)

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
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        """
        Args:
            omega (float): Controls the maximum gradient scale for small errors.
            theta (float): The threshold switching point between the log and linear zones.
            epsilon (float): Avoids division by zero and shapes the internal curve.
            alpha (float): Used in the (alpha - y) exponent to govern background sensitivity.
        """
        super(AdaptiveWingLoss2D, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

        # Precompute structural mathematical constants to save runtime FLOPs
        self.A = omega * (1.0 / (1.0 + math.pow(theta / epsilon, alpha - theta))) * \
            (alpha - theta) * (math.pow(theta / epsilon,
                                        alpha - theta - 1.0)) * (1.0 / epsilon)

        self.C = self.A * theta - omega * \
            math.log(1.0 + math.pow(theta / epsilon, alpha - theta))

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Output from the model prediction head [B, 1, H, W]
            target (torch.Tensor): Ground-truth target Hi-C matrix patch [B, 1, H, W]
        """
        # 1. Compute absolute localized spatial error
        delta_y = torch.abs(pred - target)

        # 2. Build the operating state masks based on the theta threshold
        small_error_mask = delta_y < self.theta
        large_error_mask = ~small_error_mask

        # Allocate an empty loss tensor matching our inputs
        loss = torch.zeros_like(delta_y)

        # 3. Compute Small Error Phase (Logarithmic Zone)
        # Isolate the target intensities for the small error zones
        target_small = target[small_error_mask]
        delta_y_small = delta_y[small_error_mask]

        # Calculate dynamic exponents: (alpha - y)
        pow_exponent = self.alpha - target_small

        # Run the vectorized log function
        loss[small_error_mask] = self.omega * torch.log(
            1.0 + torch.pow(delta_y_small / self.epsilon, pow_exponent)
        )

        # 4. Compute Large Error Phase (Linear Zone)
        delta_y_large = delta_y[large_error_mask]
        loss[large_error_mask] = self.A * delta_y_large - self.C

        # 5. Return the batch-averaged spatial error mean
        return loss.mean()


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        # Enforce strict matrix symmetry before computing gradients
        # This aligns the prediction with physical genomic topology
        x_sym = 0.5 * (x + x.transpose(-2, -1))

        # Calculate local differences (Gradients)
        # Delta along the genomic genomic position axis (Vertical)
        diff_h = x_sym[:, :, 1:, :] - x_sym[:, :, :-1, :]
        # Delta along the interacting genomic position axis (Horizontal)
        diff_w = x_sym[:, :, :, 1:] - x_sym[:, :, :, :-1]

        # THE CRITICAL FIXED STEP: Use the L1 Norm (.abs()) instead of L2 (.square())
        # This creates an anisotropic penalty that allows sharp step functions (TAD walls)
        h_tv = diff_h.abs().sum()
        w_tv = diff_w.abs().sum()

        # Dynamic normalization using total element counts to prevent scaling bugs
        count_h = diff_h.numel()
        count_w = diff_w.numel()

        # Average the total variation penalty across the batch size
        total_tv = (h_tv / count_h + w_tv / count_w) / b

        return total_tv


class StyleLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.criterion = nn.MSELoss()
        if weights is None:
            self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            self.weights = weights

    def gram_matrix(self, features: Tensor, mask: Tensor = None) -> torch.Tensor:
        b, c, h, w = features.shape
        if mask is not None:
            mask = F.interpolate(mask, size=(
                h, w), mode='bilinear', align_corners=False)
            features = features * mask
        features = features.view(b, c, h*w)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram /= (h*w)

        return gram

    def forward(self, pred: Tensor, y: Tensor):
        l1 = self.criterion(self.gram_matrix(pred['conv12']) / 255.0,
                            self.gram_matrix(y['conv12']) / 255.0) * self.weights[0]
        l2 = self.criterion(self.gram_matrix(pred['conv22']) / 255.0,
                            self.gram_matrix(y['conv22']) / 255.0) * self.weights[1]
        l3 = self.criterion(self.gram_matrix(pred['conv32']) / 255.0,
                            self.gram_matrix(y['conv32']) / 255.0) * self.weights[2]
        l4 = self.criterion(self.gram_matrix(pred['conv42']) / 255.0,
                            self.gram_matrix(y['conv42']) / 255.0) * self.weights[3]
        l5 = self.criterion(self.gram_matrix(pred['conv52']) / 255.0,
                            self.gram_matrix(y['conv52']) / 255.0) * self.weights[4]
        style_loss = l1 + l2 + l3 + l4 + l5
        return style_loss


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
        self.vgg_loss = VGGPerceptualLoss().to(cfg.device)
        self.style_loss = StyleLoss().to(cfg.device)
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
        corr = cov / (torch.sqrt(var_x * var_y) + 1e-12)
        return corr.mean()

    def forward(self, pred: Tensor, y: Tensor, epoch: int):
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
                loss = loss + weight * self.tv_loss(pred)
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
            else:
                raise ValueError(f"Invalid loss name: {weight_params['name']}")

        return loss
