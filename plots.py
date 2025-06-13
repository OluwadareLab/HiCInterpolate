from torch import Tensor
import matplotlib.pyplot as plt
import torch
import config

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300


def draw_real_in_out_images(x0: Tensor, x1: Tensor, x2: Tensor, pred: Tensor, epoch: int):
    num_examples = min(config.NUM_VISUALIZATION_SAMPLES, len(x1))
    x0_cpu = x0[:num_examples].cpu()
    x1_cpu = x1[:num_examples].cpu()
    pred_cpu = pred[:num_examples].cpu()
    x2_cpu = x1[:num_examples].cpu()

    plt.figure(figsize=(20, num_examples * 5))
    plt.style.use("ggplot")
    for i in range(num_examples):
        plt.subplot(num_examples, 4, i * 4 + 1)
        img_x0 = torch.clamp(x0_cpu[i].permute(1, 2, 0), 0, 1)
        plt.imshow(img_x0.numpy())
        plt.title("x0 (input)")
        plt.axis("off")

        plt.subplot(num_examples, 4, i * 4 + 2)
        true_img = torch.clamp(x1_cpu[i].permute(1, 2, 0), 0, 1)
        plt.imshow(true_img.numpy())
        plt.title("x1 (ground truth)")
        plt.axis("off")

        plt.subplot(num_examples, 4, i * 4 + 3)
        pred_img = torch.clamp(pred_cpu[i].permute(1, 2, 0), 0, 1)
        plt.imshow(pred_img.numpy())
        plt.title("prediction")
        plt.axis("off")

        plt.subplot(num_examples, 4, i * 4 + 4)
        img_x1 = torch.clamp(x2_cpu[i].permute(1, 2, 0), 0, 1)
        plt.imshow(img_x1.numpy())
        plt.title("x2 (input)")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{config.IMG_VAL_PLOT_PATH}/epoch_{epoch+1}.png")
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
