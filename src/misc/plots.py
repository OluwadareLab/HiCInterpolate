import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300
CMAP_ = mcolors.LinearSegmentedColormap.from_list(
    "juicebox", ["#FFFFFF", "#FFAAAA", "#FF5555", "#FF0000", "#B30000"], N=256
)


def draw_hic_map(num_examples, x0: np.ndarray, y: np.ndarray, pred: np.ndarray, x1: np.ndarray, file):
    data_groups = [x0, y, pred, x1]
    titles = ["$x_0$", "$y_{t=0.5}$", r"$\hat{y}_{t=0.5}$", "$x_1$"]

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
    titles = ["$y_{t=0.5}$", r"$\hat{y}_{t=0.5}$"]

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


def _plot_state_series(state, key, title, ylabel, file):
    if key not in state:
        return
    plt.figure()
    plt.plot(state[key], label=key)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.savefig(file, dpi=300, format='png')
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

    metric_specs = {
        "psnr": ("val_psnr", "PSNR on validation set", "PSNR", getattr(cfg.file, "psnr_val_plot", None)),
        "ssim": ("val_ssim", "SSIM on validation set", "SSIM", getattr(cfg.file, "ssim_val_plot", None)),
        "scc": ("val_scc", "SCC on validation set", "SCC", getattr(cfg.file, "scc_val_plot", None)),
        "hicrep": ("val_hicrep", "HiCRep on validation set", "HiCRep", getattr(cfg.file, "hicrep_val_plot", None)),
        "genome_disco": ("val_genome_disco", "GenomeDISCO on validation set", "GenomeDISCO", getattr(cfg.file, "genome_disco_val_plot", None)),
        "lpips": ("val_lpips", "LPIPS on validation set", "LPIPS", getattr(cfg.file, "lpips_val_plot", None)),
    }

    for _, (key, title, ylabel, file) in metric_specs.items():
        if file is not None:
            _plot_state_series(state, key, title, ylabel, file)

    plt.figure()
    has_metric = False
    for _, (key, _, _, _) in metric_specs.items():
        if key in state:
            plt.plot(state[key], label=key.replace("val_", "").upper())
            has_metric = True
    if has_metric:
        plt.title("validation metrics")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.legend(loc="best")
        plt.savefig(cfg.file.val_metrics_plot, dpi=300, format='png')
    plt.close()
