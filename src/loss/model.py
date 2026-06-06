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


class SymmetryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor):
        transposed = pred.transpose(-1, -2)
        diff = pred - transposed
        loss = torch.abs(diff).mean()
        return loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b = x.shape[0]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).square().sum()
        w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).square().sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


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
            torch.tensor([1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 1.0 / 1.5]),
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
        self.vgg_loss = VGGPerceptualLoss()
        self.style_loss = StyleLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.symmetry_loss = SymmetryLoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ms_ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=7, reduction="elementwise_mean"
        )
        self.to(cfg.device)

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
    def dwpc_loss(cls, pred, target):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        if pred.dim() != 3 or pred.shape != target.shape:
            raise ValueError("DWPC loss expects matching [B, N, N] or [B, 1, N, N] tensors")

        n = pred.shape[-1]
        cache_key = (n, pred.device)
        if cache_key not in cls._triu_cache:
            cls._triu_cache[cache_key] = torch.triu_indices(n, n, offset=1, device=pred.device)
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
                loss = loss + weight * self.vgg_loss(self._to_3ch(pred), self._to_3ch(y))
            elif weight_params["name"] == "style":
                loss = loss + weight * self.style_loss(self._to_3ch(pred), self._to_3ch(y))
            elif weight_params["name"] == "tv":
                loss = loss + weight * self.tv_loss(pred)
            elif weight_params["name"] == "symmetry":
                loss = loss + weight * self.symmetry_loss(pred)
            elif weight_params["name"] == "ms_ssim":
                loss = loss + weight * (1.0 - self.ms_ssim_loss(pred, y))
            elif weight_params["name"] == "dwpc":
                loss = loss + weight * (1.0 - self.dwpc_loss(pred, y))
            else:
                raise ValueError(f"Invalid loss name: {weight_params['name']}")

        return loss
