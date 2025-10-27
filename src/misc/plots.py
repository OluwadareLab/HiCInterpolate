from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 600
CMAP_ = "Reds"
# CMAP_ = "YlOrRd"


def draw_hic_map(num_examples, x0: np.ndarray, y: np.ndarray, pred: np.ndarray, x1: np.ndarray, file):
    data_groups = [x0, y, pred, x1]
    titles = ["$x_0$", "$y_{t=0.5}$", "$\hat{y}_{t=0.5}$", "$x_1$"]

    fig, axes = plt.subplots(num_examples, 4, figsize=(20, num_examples * 5))
    axes = np.atleast_2d(axes)

    for i in range(num_examples):
        for j in range(4):
            ax = axes[i, j]
            matrix = data_groups[j][i].squeeze().cpu()
            min_ = torch.min(matrix)
            max_ = torch.max(matrix)
            im = ax.imshow(matrix, cmap=CMAP_, vmin=min_, vmax=max_)
            ax.set_title(titles[j])
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{file}", dpi=300, format='png')
    plt.close()


def draw_inf_hic_map(y: np.ndarray, pred: np.ndarray, file):
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
    plt.savefig(f"{file}", dpi=300, format='png')
    plt.close()


def draw_metric(cfg, state):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(state["lr"])
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.title('learning rate')
    plt.savefig(cfg.file.lr_plot, dpi=300, format='png')
    plt.close()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(state["train_loss"], label="train loss")
    plt.plot(state["val_loss"], label="val loss")
    plt.title("loss trend")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="lower left")
    plt.savefig(cfg.file.train_val_loss_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_psnr"])
    plt.title("PSNR on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.psnr_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_ssim"])
    plt.title("SSIM on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.ssim_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_scc"])
    plt.title("SCC on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.scc_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_pcc"])
    plt.title("PCC on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.pcc_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_genome_disco"])
    plt.title("GenomeDISCO on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.genome_disco_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_ncc"])
    plt.title("NCC on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.ncc_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["val_lpips"])
    plt.title("LPIPS on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(cfg.file.lpips_val_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["grad_norms"])
    plt.title("Grad Norm During Training")
    plt.xlabel("epoch")
    plt.ylabel("grad norm")
    plt.savefig(cfg.file.grad_norm_plot, dpi=300, format='png')
    plt.close()
