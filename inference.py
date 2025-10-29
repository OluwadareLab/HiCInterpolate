import os
import numpy as np
import random
import sys
import logging
import torch
import torch.distributed as dist
import argparse
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from src.metric import eval_metrics as eval_metric
from src.misc import plots as plot
from src.inference import InfConfig, InfCustomDataset
from src import InferenceLib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


def set_seed(seed_v: int = 42):
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def get_dataloader(ds: Dataset, batch_size: int = 8, isDistributed: bool = False) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(ds, shuffle=False)
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=set_seed
        )


def reconstruct_matrix(pred_list, pad_h, pad_w, patch_size, h, w):
    pred_patches = []
    for batch in pred_list:
        for pred in batch:
            pred_patches.append(pred.squeeze(0))

    pad_size = pad_h * patch_size
    pred_padded = torch.zeros(pad_size, pad_size)
    num_patches_h = pad_h
    num_patches_w = pad_w
    i = 0
    for r in range(num_patches_h):
        for c in range(num_patches_w):
            pred_padded[
                r * patch_size:(r + 1) * patch_size,
                c * patch_size:(c + 1) * patch_size
            ] = pred_patches[i]
            i += 1

    pred_matrix = pred_padded[:h, :w]

    return pred_matrix


def main(config_filename: str, isDistributed: bool = False):
    yaml_cfg = OmegaConf.load(f"./src/inference/{config_filename}.yaml")
    structured_cfg = OmegaConf.structured(InfConfig)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)
    # OmegaConf.update(cfg, "dir.model_state", model_state_dir)

    log = base_logger(cfg.file.log)
    if isDistributed:
        ddp_setup()

    batch_size = cfg.data.batch_size
    device = cfg.device
    if os.path.exists(cfg.file.model):
        cds = InfCustomDataset(record_file=cfg.file.inference, img_dir=cfg.dir.image,
                               img_map=cfg.data.interpolator_images_map)
        ds = cds._get_inference_dl()
        dl = get_dataloader(
            ds=ds, batch_size=batch_size, isDistributed=isDistributed)
        inference = InferenceLib.HiCInterpolate(
            cfg=cfg, log=log, model=cfg.file.model, dl=dl, isDistributed=isDistributed)
        inference._inference()

        pred_list = inference._get_prediction()
        patch_size = cfg.data.patch
        path = cfg.file.inference_raw
        with open(path, "r") as file:
            original_path = [line.rstrip() for line in file]

        y = np.load(f"{original_path[0]}/img_2.npy")

        h, w = y.shape
        _extension = math.ceil(h/patch_size)
        pad_h = _extension
        pad_w = _extension
        pred = reconstruct_matrix(
            pred_list, pad_h, pad_w, patch_size, h, w)

        y = torch.from_numpy(y).to(device).unsqueeze(0).unsqueeze(0).float()
        pred = pred.to(device).unsqueeze(0).unsqueeze(0).float()

        print(f"Ploating heatmap...")
        plot.draw_inf_hic_map(y=y, pred=pred, file=f"{cfg.file.hic_map}")
        psnr = eval_metric.get_psnr(pred, y).item()
        ssim = eval_metric.get_ssim(pred, y).item()
        scc = eval_metric.get_scc(pred, y).item()
        pcc = eval_metric.get_pcc(pred, y).item()
        genome_disco = eval_metric.get_genome_disco(pred, y).item()
        ncc = eval_metric.get_ncc(pred, y).item()
        lpips = eval_metric.get_lpips(pred, y).item()

        scores = f"PSNR: {format(psnr, '.4f')}, SSIM: {format(ssim, '.4f')}, PCC: {format(pcc, '.4f')}, SCC: {format(scc, '.4f')}, GenomeDISCO: {format(genome_disco, '.4f')}, NCC: {format(ncc, '.4f')}, LPIPS: {format(lpips, '.4f')};"

        metric_file = cfg.file.metrics
        os.makedirs(os.path.dirname(metric_file), exist_ok=True)
        with open(metric_file, "a") as file:
            file.write(scores+"\n")
            file.close()

        log.info(f"{scores}")
        print(f"[INFO] {scores}")

    if isDistributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)
    print("sys.argv:", sys.argv)
    parser = argparse.ArgumentParser(
        description='ap film distributed training job')
    parser.add_argument('-dis', '--distributed', dest="distributed",
                        action='store_true', help='Distributed training (default: False)')
    parser.add_argument('-cfg', '--config', dest="config",  type=str, default="config",
                        help='Configuration filename without extension. This file should be in the configs folder (default: config)')
    args = parser.parse_args()

    # main(args.config, args.distributed)
    main(config_filename="config", isDistributed=False)
