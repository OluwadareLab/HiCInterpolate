
import sys
import os
import numpy as np
import scipy.io as sio
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.nn import L1Loss, MSELoss, Module, functional as F, ModuleList, MaxPool2d
from torch import Tensor
from typing import Tuple, Dict
from src.misc import metrics as metric
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class _L1Loss(Module):
    def __init__(self):
        super().__init__()
        self.criterion = L1Loss()

    def forward(self, pred: Tensor, y: Tensor):
        loss = self.criterion(pred, y)
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
        self.criterion = L1Loss()

    def forward(self, pred: Tensor):
        transposed = pred.transpose(-1, -2)
        loss = self.criterion(pred, transposed)
        return loss


class SSIMLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, y: Tensor):
        loss = 1-metric.calculate_ssim(pred, y)
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


class GenomeDISCOLoss(Module):
    def __init__(self):
        super().__init__()

    def _to_transition_torch(self, m):
        row_sum = m.sum(dim=-1, keepdim=True)
        row_sum = torch.clamp(row_sum, min=1e-12)
        return m / row_sum

    def _compute_reproducibility_gpu(self, m1, m2, transition=True, tmin=3, tmax=3):
        assert m1.shape == m2.shape, "Input shapes must match"
        B, C, H, W = m1.shape
        assert H == W, "Matrix must be square"
        m1_sym = m1 + m1.transpose(-1, -2)
        m2_sym = m2 + m2.transpose(-1, -2)

        if transition:
            m1_sym = self._to_transition_torch(m1_sym)
            m2_sym = self._to_transition_torch(m2_sym)

        row_sum1 = m1_sym.sum(dim=-1)
        row_sum2 = m2_sym.sum(dim=-1)
        nonzero1 = (row_sum1 > 0).float().sum(dim=-1)
        nonzero2 = (row_sum2 > 0).float().sum(dim=-1)
        nonzero_total = 0.5 * (nonzero1 + nonzero2)
        nonzero_total = torch.clamp(nonzero_total, min=1e-8)

        rw1 = m1_sym.clone()
        rw2 = m2_sym.clone()

        scores = []

        for t in range(1, tmax + 1):
            if t > 1:
                rw1 = torch.matmul(rw1, m1_sym)
                rw2 = torch.matmul(rw2, m2_sym)

            if t >= tmin:
                diff = torch.abs(rw1 - rw2).sum(dim=(-2, -1))
                score_t = diff / nonzero_total
                scores.append(score_t)

        scores = torch.stack(scores, dim=-1)
        if tmin == tmax:
            auc = scores[..., 0]
            auc = torch.clamp(auc, 0, 2)
        else:
            f_i = scores[..., :-1]
            f_ip1 = scores[..., 1:]
            auc = 0.5 * (f_i + f_ip1)
            auc = auc.sum(dim=-1) / (scores.shape[-1] - 1)

        reproducibility = 1.0 - auc
        reproducibility = reproducibility.mean(dim=1)

        return reproducibility

    def forward(self, pred: Tensor, y: Tensor):
        reproducibility = self._compute_reproducibility_gpu(pred, y)
        mean_repro = torch.mean(reproducibility)
        loss = 1.0-mean_repro
        return loss


# class _VGG(Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         vgg = sio.loadmat(self.cfg.file.vgg_model)
#         self.vgg_layers = vgg['layers'][0]

#     def layer_(self, name, input_tensor, weight=None, bias=None, stride=1):
#         if name == "conv":
#             pred = F.conv2d(input=input_tensor, weight=weight, bias=bias,
#                             stride=stride, padding='same')
#             pred = F.relu(input=pred, inplace=True)
#         elif name == "pool":
#             pred = F.avg_pool2d(input=input_tensor,
#                                 kernel_size=2, stride=2, padding=1)
#         else:
#             raise NotImplemented

#         return pred

#     def get_weights_bias(self, vgg_layers: np.ndarray, index: int, device) -> Tuple[Tensor, Tensor]:
#         weights = vgg_layers[index][0][0][2][0][0]
#         weights = np.transpose(weights, (3, 2, 0, 1))
#         weights = torch.from_numpy(weights).to(
#             device, dtype=torch.float32)

#         bias = vgg_layers[index][0][0][2][0][1]
#         bias = torch.from_numpy(np.reshape(
#             bias, (bias.size))).to(device, dtype=torch.float32)

#         return weights, bias

#     def forward(self, input: Tensor) -> Dict[str, Tensor]:
#         psudo_input = input.repeat(1, 3, 1, 1)
#         mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32,
#                             device=input.device).view(1, 3, 1, 1)
#         std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32,
#                            device=input.device).view(1, 3, 1, 1)
#         img_norm = (psudo_input - mean) / std

#         net = {}
#         net['input'] = img_norm
#         weights, bias = self.get_weights_bias(self.vgg_layers, 0, input.device)
#         net['conv11'] = self.layer_(
#             name="conv", input_tensor=net['input'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(self.vgg_layers, 2, input.device)
#         net['conv12'] = self.layer_(
#             name="conv", input_tensor=net['conv11'], weight=weights, bias=bias)
#         net['pool1'] = self.layer_(name="pool", input_tensor=net['conv11'])

#         weights, bias = self.get_weights_bias(self.vgg_layers, 5, input.device)
#         net['conv21'] = self.layer_(
#             name="conv", input_tensor=net['pool1'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(self.vgg_layers, 7, input.device)
#         net['conv22'] = self.layer_(
#             name="conv", input_tensor=net['conv21'], weight=weights, bias=bias)
#         net['pool2'] = self.layer_(name="pool", input_tensor=net['conv21'])

#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 10, input.device)
#         net['conv31'] = self.layer_(
#             name="conv", input_tensor=net['pool2'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 12, input.device)
#         net['conv32'] = self.layer_(
#             name="conv", input_tensor=net['conv31'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 14, input.device)
#         net['conv33'] = self.layer_(
#             name="conv", input_tensor=net['conv32'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 16, input.device)
#         net['conv34'] = self.layer_(
#             name="conv", input_tensor=net['conv33'], weight=weights, bias=bias)
#         net['pool3'] = self.layer_(name="pool", input_tensor=net['conv34'])

#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 19, input.device)
#         net['conv41'] = self.layer_(
#             name="conv", input_tensor=net['pool3'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 21, input.device)
#         net['conv42'] = self.layer_(
#             name="conv", input_tensor=net['conv41'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 23, input.device)
#         net['conv43'] = self.layer_(
#             name="conv", input_tensor=net['conv42'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 25, input.device)
#         net['conv44'] = self.layer_(
#             name="conv", input_tensor=net['conv43'], weight=weights, bias=bias)
#         net['pool4'] = self.layer_(name="pool", input_tensor=net['conv44'])

#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 28, input.device)
#         net['conv51'] = self.layer_(
#             name="conv", input_tensor=net['pool4'], weight=weights, bias=bias)
#         weights, bias = self.get_weights_bias(
#             self.vgg_layers, 30, input.device)
#         net['conv52'] = self.layer_(
#             name="conv", input_tensor=net['conv51'], weight=weights, bias=bias)

#         return net


# class VGGLoss(Module):
#     def __init__(self, weights=None):
#         super().__init__()
#         self.criterion = MSELoss()
#         if weights is None:
#             self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
#         else:
#             self.weights = weights

#     def forward(self, pred: Tensor, y: Tensor):
#         l1 = self.criterion(pred['conv12']/255.0,
#                             y['conv12']/255.0) * self.weights[0]
#         l2 = self.criterion(pred['conv22']/255.0,
#                             y['conv22']/255.0) * self.weights[1]
#         l3 = self.criterion(pred['conv32']/255.0,
#                             y['conv32']/255.0) * self.weights[2]
#         l4 = self.criterion(pred['conv42']/255.0,
#                             y['conv42']/255.0) * self.weights[3]
#         l5 = self.criterion(pred['conv52']/255.0,
#                             y['conv52']/255.0) * self.weights[4]
#         vgg_loss = l1 + l2 + l3 + l4 + l5
#         return vgg_loss


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
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGPerceptualLoss(nn.Module):
    def __init__(self, cfg):
        super(VGGPerceptualLoss, self).__init__()
        self.cfg = cfg
        self.vgg_pretrained_features = models.vgg19(
            pretrained=True).features.to(self.cfg.device)
        self.normalize = MeanShift([0.485, 0.456, 0.406], [
                                   0.229, 0.224, 0.225], norm=True).to(self.cfg.device)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        X = self.normalize(pred)
        Y = self.normalize(target)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean()
                k += 1
        return loss


class CombinedLoss(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.vgg = _VGG(self.cfg)
        self.vgg_loss = VGGPerceptualLoss(self.cfg)
        self.style_loss = StyleLoss()
        self.l1_loss = _L1Loss()
        self.mse_loss = _MSELoss()
        self.charbonnier_loss = CharbonnierLoss()
        self.tv_loss = TVLoss()
        self.symmetry_loss = SymmetryLoss()
        self.ssim_loss = SSIMLoss()
        self.genome_disco_loss = GenomeDISCOLoss()

    def weight_schedule(self, weight_params: tuple, epoch: int) -> float:
        i = 0
        limit = len(weight_params["boundaries"])
        while i < limit:
            if epoch >= weight_params["boundaries"][i]:
                return weight_params["values"][i+1]
            i += 1
        return weight_params["values"][0]

    def forward(self, pred: Tensor, y: Tensor, epoch: int):
        loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        for weight_params in self.cfg.loss.weight_parameters:
            weight = self.weight_schedule(
                weight_params=weight_params, epoch=epoch)

            if (weight_params["name"] == "vgg" and weight > 0.0) or (weight_params["name"] == "style" and weight > 0.0):
                # vgg_pred = self.vgg(pred*255)
                # vgg_y = self.vgg(y*255)
                vgg_pred = pred.repeat(1, 3, 1, 1)
                vgg_y = y.repeat(1, 3, 1, 1)

            if weight_params["name"] == "l1" and weight > 0.0:
                l1_loss = self.l1_loss(pred, y)
                l1_loss = l1_loss * weight
                loss += l1_loss
            elif weight_params["name"] == "mse" and weight > 0.0:
                mse_loss = self.mse_loss(pred, y)
                mse_loss = mse_loss * weight
                loss += mse_loss
            elif weight_params["name"] == "charbonnier" and weight > 0.0:
                charbonnier_loss = self.charbonnier_loss(pred, y)
                charbonnier_loss = charbonnier_loss * weight
                loss += charbonnier_loss
            elif weight_params["name"] == "ssim" and weight > 0.0:
                ssim_loss = self.ssim_loss(pred, y)
                ssim_loss = ssim_loss * weight
                loss += ssim_loss
            elif weight_params["name"] == "vgg" and weight > 0.0:
                vgg_loss = self.vgg_loss(vgg_pred, vgg_y)
                vgg_loss = vgg_loss * weight
                loss += vgg_loss
            elif weight_params["name"] == "style" and weight > 0.0:
                style_loss = self.style_loss(vgg_pred, vgg_y)
                style_loss = style_loss * weight
                loss += style_loss
            elif weight_params["name"] == "tv" and weight > 0.0:
                tv_loss = self.tv_loss(pred)
                tv_loss = tv_loss * weight
                loss += tv_loss
            elif weight_params["name"] == "symmetry" and weight > 0.0:
                symmetry_loss = self.symmetry_loss(pred)
                symmetry_loss = symmetry_loss * weight
                loss += symmetry_loss
            elif weight_params["name"] == "genome_disco" and weight > 0.0:
                genome_disco_loss = self.genome_disco_loss(pred, y)
                genome_disco_loss = genome_disco_loss * weight
                loss += genome_disco_loss
        return loss
