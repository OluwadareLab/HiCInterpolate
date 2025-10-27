import random
import torch
import numpy as np
import torch.nn as nn
import sys
import os
import logging
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import evaluate
from load_data import CustomDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import UNet
import plots as plot

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_logger(filename):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


def set_seed(seed_v: int = 42):
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def get_dataloader(ds: Dataset, batch_size: int = 8, shuffle: bool = False, isDistributed: bool = False) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(ds, shuffle=shuffle)
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=set_seed
        )


def train_model(
        log,
        model,
        device,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    dataset_dict = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/triplets/kr_log_clip_norm_diag/256/dataset_dict.txt"
    img_dir = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/triplets/kr_log_clip_norm_diag/256"
    img_map = {
        "frame_0": "img1.npy",
        "frame_1": "img2.npy",
        "frame_2": "img3.npy"
    }
    train_val_test_ratio = [0.10, 0.0125, 0.0125]
    isDistributed = False

    cds = CustomDataset(record_file=dataset_dict, img_dir=img_dir,
                        img_map=img_map, shuffle=True, train_val_test_ratio=train_val_test_ratio)
    train_dataset, val_dataset, _ = cds._get_dataset()

    train_dataloader = get_dataloader(
        ds=train_dataset, batch_size=batch_size, shuffle=True, isDistributed=isDistributed)
    val_dataloader = get_dataloader(ds=val_dataset, batch_size=batch_size,
                                    shuffle=True, isDistributed=isDistributed)

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    train_loss_list = []
    eval_loss_list = []
    eval_ssim_list = []
    eval_psnr_list = []
    grad_norm_list = []

    patience = 10
    best_metric = -float('inf')
    epochs_no_improve = 0
    best_model_path = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/flow/unet_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        batch_loss_list = []
        batch_grad_norm_list = []
        for _, (x1, y, x3, _) in enumerate(tqdm(train_dataloader)):
            x1 = x1.to(device=device, dtype=torch.float32,
                       memory_format=torch.channels_last)
            y = y.to(device=device, dtype=torch.float32,
                     memory_format=torch.channels_last)
            x3 = x3.to(device=device, dtype=torch.float32,
                       memory_format=torch.channels_last)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                pred = model(x1, x3)
                loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            batch_grad_norm_list.append(total_norm)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            batch_loss_list.append(loss)

        mean_train_loss = torch.stack(batch_loss_list).mean(
        ) if batch_loss_list else torch.tensor(0.0, device=device)
        mean_train_loss = mean_train_loss.item()
        train_loss_list.append(mean_train_loss)

        mean_grad_norm = sum(
            batch_grad_norm_list)/len(batch_grad_norm_list) if batch_grad_norm_list else 0.0
        grad_norm_list.append(mean_grad_norm)

        eval_loss, eval_ssim, eval_psnr = evaluate(
            model, val_dataloader, device, amp, epoch)
        eval_loss = eval_loss.item()
        eval_loss_list.append(eval_loss)

        eval_ssim = eval_ssim.item()
        eval_ssim_list.append(eval_ssim)
        eval_psnr = eval_psnr.item()
        eval_psnr_list.append(eval_psnr)

        plot.draw_loss(train_loss_list, eval_loss_list,
                       "Train & Evaluation Loss", "loss", "unet_train_eval_loss")
        plot.draw_metric(eval_ssim_list, "SSIM", "index", "unet_ssim")
        plot.draw_metric(eval_psnr_list, "PSNR", "dB", "unet_psnr")
        plot.draw_metric(grad_norm_list, "Gradient Norm",
                         "L2 Norm", "unet_grad_norm")

        scheduler.step(eval_loss)

        msg = f"[Epoch {epoch}/{epochs}] Grad Norm: {mean_grad_norm:.4f}; Train Loss: {mean_train_loss:.6f}; Eval Loss: {eval_loss:.6f}; PSNR: {eval_psnr:.4f}; SSIM: {eval_ssim:.4f}"
        log.info(msg)
        print(msg)

        if (eval_psnr + eval_ssim)/2 > best_metric:
            best_metric = (eval_psnr + eval_ssim)/2
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': eval_psnr,
                'ssim': eval_ssim,
                'loss': eval_loss
            }, best_model_path)
            msg = f"New best model saved!"
            log.info(msg)
            print(msg)
        elif epoch > 200:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            msg = f"[Epoch {epoch}/{epochs}] Early stopping triggered after {patience} epochs without improvement."
            log.info(msg)
            print(msg)
            break


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1, bilinear=True)
    model = model.to(memory_format=torch.channels_last)
    log = get_logger(
        "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/unet/flow/unet.log")
    model.to(device=device)
    try:
        train_model(
            log=log,
            model=model,
            epochs=1000,
            batch_size=8,
            learning_rate=1e-5,
            device=device,
            amp=True
        )
    except torch.cuda.OutOfMemoryError:
        print("[ERR] Cuda out of memory!")
