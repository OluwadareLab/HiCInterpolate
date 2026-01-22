import os
import numpy as np
import random
import csv
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
from scipy.ndimage import gaussian_filter as sp_gf
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


def main(config_filename: str, isDistributed: bool = False):
    yaml_cfg = OmegaConf.load(f"./src/inference/{config_filename}.yaml")
    structured_cfg = OmegaConf.structured(InfConfig)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)

    log = base_logger(cfg.file.log)
    if isDistributed:
        ddp_setup()

    batch_size = cfg.data.batch_size
    if os.path.exists(cfg.file.model):
        cds = InfCustomDataset(record_file=cfg.file.inference,
                               img_dir=cfg.dir.image, img_map=cfg.data.interpolator_images_map)
        ds = cds._get_inference_dl()
        dl = get_dataloader(ds=ds, batch_size=batch_size,
                            isDistributed=isDistributed)
        inference = InferenceLib.HiCInterpolate(
            cfg=cfg, log=log, model=cfg.file.model, dl=dl, isDistributed=isDistributed)
        inference._inference()

        pred_list = inference._get_prediction()
        numpy_list = [t.cpu().numpy() for t in pred_list]
        inferenced_filename = f"{cfg.dir.output}/inferenced.npy"
        np.save(inferenced_filename, np.array(numpy_list, dtype=object))
        print(f"Infeenced file save at {inferenced_filename}")

    if isDistributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(
        description='ap film distributed training job')
    parser.add_argument('-cfg', '--config', dest="config",  type=str, default="config",
                        help='Configuration filename without extension. This file should be in the configs folder (default: config)')
    args = parser.parse_args()

    main(args.config)