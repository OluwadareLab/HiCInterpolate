
import sys
import os
import torch
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, Module, functional as F
from torch import Tensor
from src.metric import eval_metrics as eval_metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class _L1Loss(Module):
    def __init__(self):
        super().__init__()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, pred: Tensor, y: Tensor):
        loss = self.huber_loss(pred, y)
        return loss


class _MSELoss(Module):
    def __init__(self):
        super().__init__()
        self.criterion = MSELoss()

    def forward(self, pred: Tensor, y: Tensor):
        loss = self.criterion(pred, y)
        return loss


class CharbonnierLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, y: Tensor, epsilon=1e-3):
        diff = pred - y
        loss = torch.mean(torch.sqrt(diff ** 2 + epsilon ** 2))
        return loss


class SymmetryLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor):
        transposed = pred.transpose(-1, -2)
        diff = pred - transposed
        loss = torch.abs(diff).mean()
        return loss


class SSIMLoss(Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = StructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=11
        )
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, pred, target):
        l_ssim = 1.0 - self.ssim_loss(pred, target)
        return l_ssim


class FlowSmoothnessLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, flow: Tensor):
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        loss = torch.mean(dx) + torch.mean(dy)
        return loss


class TVLoss(Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class StyleLoss(Module):
    def __init__(self, weights=None):
        super().__init__()
        self.criterion = MSELoss()
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

        self.weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 1.0 / 1.5]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        self.eval()
        X = self.normalize(pred)
        Y = self.normalize(target)

        loss = 0.0

        X = self.slice1(X)
        Y = self.slice1(Y)
        loss += self.weights[0] * nn.functional.l1_loss(X, Y.detach())

        X = self.slice2(X)
        Y = self.slice2(Y)
        loss += self.weights[1] * nn.functional.l1_loss(X, Y.detach())

        X = self.slice3(X)
        Y = self.slice3(Y)
        loss += self.weights[2] * nn.functional.l1_loss(X, Y.detach())

        X = self.slice4(X)
        Y = self.slice4(Y)
        loss += self.weights[3] * nn.functional.l1_loss(X, Y.detach())

        X = self.slice5(X)
        Y = self.slice5(Y)
        loss += self.weights[4] * nn.functional.l1_loss(X, Y.detach())

        return loss


class CombinedLoss(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vgg_loss = VGGPerceptualLoss().to(self.cfg.device)
        self.style_loss = StyleLoss().to(self.cfg.device)
        self.l1_loss = _L1Loss().to(self.cfg.device)
        self.mse_loss = _MSELoss().to(self.cfg.device)
        self.charbonnier_loss = CharbonnierLoss().to(self.cfg.device)
        self.tv_loss = TVLoss().to(self.cfg.device)
        self.symmetry_loss = SymmetryLoss().to(self.cfg.device)
        self.ssim_loss = SSIMLoss().to(self.cfg.device)
        self.flow_smoothness_loss = FlowSmoothnessLoss().to(self.cfg.device)

    def weight_schedule(self, weight_params: tuple, epoch: int) -> float:
        for i, boundary in enumerate(weight_params["boundaries"]):
            if epoch < boundary:
                return weight_params["values"][i]
        return weight_params["values"][-1]

    def forward(self, pred: Tensor, y: Tensor, epoch: int, forward_flow: Tensor = None, backward_flow: Tensor = None):
        loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        for weight_params in self.cfg.loss.weight_parameters:
            weight = self.weight_schedule(
                weight_params=weight_params, epoch=epoch)
            if weight <= 0.0:
                continue

            if weight_params["name"] == "l1":
                l1_loss = self.l1_loss(pred, y)
                l1_loss = l1_loss * weight
                loss += l1_loss
            elif weight_params["name"] == "mse":
                mse_loss = self.mse_loss(pred, y)
                mse_loss = mse_loss * weight
                loss += mse_loss
            elif weight_params["name"] == "charbonnier":
                charbonnier_loss = self.charbonnier_loss(pred, y)
                charbonnier_loss = charbonnier_loss * weight
                loss += charbonnier_loss
            elif weight_params["name"] == "ssim":
                ssim_loss = self.ssim_loss(pred, y)
                ssim_loss = ssim_loss * weight
                loss += ssim_loss
            elif weight_params["name"] == "vgg":
                vgg_pred = pred.clamp(0, 1).repeat(
                    1, 3, 1, 1) if pred.shape[1] == 1 else pred.clamp(0, 1)
                vgg_y = y.clamp(0, 1).repeat(
                    1, 3, 1, 1) if y.shape[1] == 1 else y.clamp(0, 1)
                vgg_loss = self.vgg_loss(vgg_pred, vgg_y)
                vgg_loss = vgg_loss * weight
                loss += vgg_loss
            elif weight_params["name"] == "style":
                vgg_pred = pred.clamp(0, 1).repeat(
                    1, 3, 1, 1) if pred.shape[1] == 1 else pred.clamp(0, 1)
                vgg_y = y.clamp(0, 1).repeat(
                    1, 3, 1, 1) if y.shape[1] == 1 else y.clamp(0, 1)
                style_loss = self.style_loss(vgg_pred, vgg_y)
                style_loss = style_loss * weight
                loss += style_loss
            elif weight_params["name"] == "tv":
                tv_loss = self.tv_loss(pred)
                tv_loss = tv_loss * weight
                loss += tv_loss
            elif weight_params["name"] == "flow_smoothness":
                if forward_flow is None or backward_flow is None:
                    continue

                fwd_flows = forward_flow if isinstance(
                    forward_flow, list) else [forward_flow]
                bwd_flows = backward_flow if isinstance(
                    backward_flow, list) else [backward_flow]
                flow_smooth_loss = torch.tensor(
                    0.0, device=pred.device, dtype=torch.float32)
                for fwd_f, bwd_f in zip(fwd_flows, bwd_flows):
                    flow_smooth_loss += self.flow_smoothness_loss(
                        fwd_f) + self.flow_smoothness_loss(bwd_f)
                flow_smooth_loss = flow_smooth_loss * weight
                loss += flow_smooth_loss
        return loss
