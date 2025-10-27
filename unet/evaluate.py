
import sys
import os
import torch
import torch.nn.functional as F
import eval_metrics as eval_metric
import plots as plot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, epoch):
    net.eval()
    loss_list = []
    ssim_list = []
    psnr_list = []
    drawn = 0
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for _, (x1, y, x3, _) in enumerate(dataloader):
            x1 = x1.to(device=device, dtype=torch.float32,
                       memory_format=torch.channels_last)
            y = y.to(device=device, dtype=torch.float32,
                     memory_format=torch.channels_last)
            x3 = x3.to(device=device, dtype=torch.float32,
                       memory_format=torch.channels_last)
            pred = net(x1, x3)
            loss_list.append(F.mse_loss(
                input=pred, target=y, reduction='mean'))
            ssim_list.append(eval_metric.get_ssim(pred=pred, target=y))
            psnr_list.append(eval_metric.get_psnr(pred=pred, target=y))

            if drawn == 3:
                plot.draw_hic_map(y=y, pred=pred,
                                  filename=f"unet_hic_map_{epoch}")
            drawn += 1

    mean_loss = torch.stack(loss_list).mean(
    ) if loss_list else torch.tensor(0.0, device=device)
    mean_ssim = torch.stack(ssim_list).mean(
    ) if ssim_list else torch.tensor(0.0, device=device)
    mean_psnr = torch.stack(psnr_list).mean(
    ) if psnr_list else torch.tensor(0.0, device=device)

    net.train()
    return mean_loss, mean_ssim, mean_psnr
