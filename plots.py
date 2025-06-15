from torch import Tensor
import matplotlib.pyplot as plt
import torch
import config
import seaborn as sns
import numpy as np

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300


def draw_real_in_out_images(x0: Tensor, y: Tensor, pred: Tensor, x1: Tensor,  epoch: int):
    num_examples = min(config.NUM_VISUALIZATION_SAMPLES, len(y))
    x0_cpu = x0[:num_examples].cpu()
    y_cpu = y[:num_examples].cpu()
    pred_cpu = pred[:num_examples].cpu()
    x1_cpu = x1[:num_examples].cpu()

    plt.figure(figsize=(20, num_examples * 5))
    plt.style.use("ggplot")
    for i in range(num_examples):
        plt.subplot(num_examples, 4, i * 4 + 1)
        log_matrix = np.log10(x0_cpu[i].squeeze(0) + 1)
        sns.heatmap(x0_cpu[i].squeeze(0), cmap="YlOrRd_r")
        plt.title("x0")
        plt.xlabel("Genome coordinates")
        plt.ylabel("Genome coordinates")
        plt.axis("off")
        plt.tight_layout()

        plt.subplot(num_examples, 4, i * 4 + 2)
        log_matrix = np.log10(y_cpu[i].squeeze(0) + 1)
        sns.heatmap(y_cpu[i].squeeze(0), cmap="YlOrRd_r")
        plt.title("y")
        plt.xlabel("Genome coordinates")
        plt.ylabel("Genome coordinates")
        plt.axis("off")
        plt.tight_layout()

        plt.subplot(num_examples, 4, i * 4 + 3)
        log_matrix = np.log10(pred_cpu[i].squeeze(0) + 1)
        sns.heatmap(pred_cpu[i].squeeze(0), cmap="YlOrRd_r")
        plt.title("pred")
        plt.xlabel("Genome coordinates")
        plt.ylabel("Genome coordinates")
        plt.axis("off")
        plt.tight_layout()

        plt.subplot(num_examples, 4, i * 4 + 4)
        log_matrix = np.log10(x1_cpu[i].squeeze(0) + 1)
        sns.heatmap(x1_cpu[i].squeeze(0), cmap="YlOrRd_r")
        plt.title("x1")
        plt.xlabel("Genome coordinates")
        plt.ylabel("Genome coordinates")
        plt.axis("off")
        plt.tight_layout()

    plt.savefig(f"{config.IMG_VAL_PLOT_PATH}/epoch_{epoch+1}.png", )
    plt.close()


def draw_metric(state):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(state["lr"])
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.title('learning rate')
    plt.savefig(config.LR_FILE)
    plt.close()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(state["train_loss"], label="train loss")
    plt.plot(state["val_loss"], label="val loss")
    plt.title("loss trend")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="lower left")
    plt.savefig(config.TRAIN_VAL_PLOT_FILE)
    plt.close()

    plt.figure()
    plt.plot(state["val_ssim"])
    plt.title("ssim on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(config.SSIM_EVAL_PLOT_FILE)
    plt.close()

    plt.figure()
    plt.plot(state["val_psnr"])
    plt.title("psnr on validation set")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(config.PSNR_EVAL_PLOT_FILE)
    plt.close()
