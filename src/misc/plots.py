from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300
CMAP_ = mcolors.LinearSegmentedColormap.from_list(
    "juicebox",
    [
        "#fee8c8",
        "#fdbb84",
        "#e34a33",
        "#b30000"
    ],
    N=256
)


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
        matrix = data_groups[i]

        min_ = np.min(matrix)
        max_ = np.max(matrix)

        im = ax.imshow(matrix, cmap=CMAP_, vmin=min_, vmax=max_)
        ax.set_title(titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{file}.png", dpi=300, format='png')
    plt.close()


def draw_metric(cfg, state):
    plt.figure()
    plt.plot(state["lr"])
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.title('learning rate')
    plt.savefig(cfg.file.lr_plot, dpi=300, format='png')
    plt.close()

    plt.figure()
    plt.plot(state["train_loss"], label="train loss")
    plt.plot(state["val_loss"], label="val loss")
    plt.title("loss trend")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig(cfg.file.train_val_loss_plot, dpi=300, format='png')
    plt.close()

    plt.style.use("ggplot")
    plt.figure()
    if "val_sparse_f1" in state:
        plt.plot(state["val_sparse_f1"], label="SparseF1")
    if "val_score" in state:
        plt.plot(state["val_score"], label="SparseScore")
    plt.title("sparse validation metrics")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend(loc="upper left")
    plt.savefig(cfg.file.val_metrics_plot, dpi=300, format='png')
    plt.close()

    # plt.figure()
    # plt.plot(state["val_psnr"])
    # plt.title("PSNR on validation set")
    # plt.xlabel("epoch")
    # plt.ylabel("PSNR")
    # plt.savefig(cfg.file.psnr_val_plot, dpi=300, format='png')
    # plt.close()


    if "val_sparse_f1" in state:
        plt.figure()
        plt.plot(state["val_sparse_precision"], label="Precision")
        plt.plot(state["val_sparse_recall"], label="Recall")
        plt.plot(state["val_sparse_f1"], label="F1")
        plt.title("Sparse support metrics")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.legend(loc="lower right")
        plt.savefig(f"{cfg.dir.output}/{cfg.model.name}_sparse_support_plot.png", dpi=300, format='png')
        plt.close()

    if "val_pred_density" in state:
        plt.figure()
        plt.plot(state["val_pred_density"], label="PredDensity")
        plt.plot(state["val_target_density"], label="TargetDensity")
        plt.plot(state["val_density_error"], label="DensityError")
        plt.title("Sparse density")
        plt.xlabel("epoch")
        plt.ylabel("fraction")
        plt.legend(loc="upper right")
        plt.savefig(f"{cfg.dir.output}/{cfg.model.name}_density_plot.png", dpi=300, format='png')
        plt.close()

    if "val_nonzero_mae" in state:
        plt.figure()
        plt.plot(state["val_nonzero_mae"], label="NonzeroMAE")
        plt.plot(state["val_zero_mae"], label="ZeroMAE")
        plt.title("Sparse intensity error")
        plt.xlabel("epoch")
        plt.ylabel("MAE")
        plt.legend(loc="upper right")
        plt.savefig(f"{cfg.dir.output}/{cfg.model.name}_sparse_mae_plot.png", dpi=300, format='png')
        plt.close()

    # plt.figure()
    # plt.plot(state["val_lpips"])
    # plt.title("LPIPS on validation set")
    # plt.xlabel("epoch")
    # plt.ylabel("LPIPS")
    # plt.savefig(cfg.file.lpips_val_plot, dpi=300, format='png')
    # plt.close()
