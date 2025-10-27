import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['figure.dpi'] = 300
CMAP_ = "Reds"


def draw_hic_map(y: Tensor, pred: Tensor, filename, num_exp=2):
    data_groups = [y, pred]
    titles = ["$y_{t=0.5}$", "$\hat{y}_{t=0.5}$"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = np.atleast_2d(axes)

    for i in range(len(data_groups)):
        ax = axes[0, i]
        matrix = data_groups[i][0].squeeze().cpu()
        min_ = torch.min(matrix)
        max_ = torch.max(matrix)
        im = ax.imshow(matrix, cmap=CMAP_, vmin=min_, vmax=max_)
        ax.set_title(titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(
        f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/flow/hic_map/{filename}.png", dpi=300, format='png')
    plt.close()


def draw_loss(train_loss, eval_loss, title, ylabel, filename):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(eval_loss, label="val")
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(
        f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/flow/{filename}.png", dpi=300, format='png')
    plt.close()


def draw_metric(metric, title, ylabel, filename):
    plt.figure()
    plt.plot(metric)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(
        f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/flow/{filename}.png", dpi=300, format='png')
    plt.close()
