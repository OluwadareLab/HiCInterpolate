import random
import numpy as np
import sys
import os
import logging
import torch
import torch.distributed as dist
from src import TrainLib, CustomDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple
from omegaconf import OmegaConf
from configs.config import Config


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


def prepare_dataloader(train_ds: Dataset, val_ds: Dataset, batch_size: int = 0, isDistributed: bool = False) -> Tuple[Dataset, Dataset]:
    if isDistributed:
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(train_ds, shuffle=True)
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(val_ds, shuffle=False)
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_seed
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False
        )
    return train_dl, val_dl


def main(config_filename: str, isDistributed: bool = False, load_snapshot: bool = False):

    yaml_cfg = OmegaConf.load(f"./configs/{config_filename}.yaml")
    structured_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    output_path = f"{cfg.paths.root_dir}/output/{config_filename}"
    model_state_dir = f"{cfg.paths.root_dir}/models/{config_filename}"
    os.makedirs(f"{output_path}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "paths.output_dir", output_path)
    OmegaConf.update(cfg, "paths.model_state_dir", model_state_dir)

    log = base_logger(cfg.paths.log_file)
    if isDistributed:
        ddp_setup()
    cds = CustomDataset(config=cfg)
    train_ds, val_ds = cds.get_train_val_dl()
    train_dl, val_dl = prepare_dataloader(
        train_ds, val_ds, cfg.data.batch_size, isDistributed)
    trainer = TrainLib.Trainer(cfg, log, train_dl, val_dl,
                               load_snapshot, isDistributed)
    trainer.train(cfg.training.epochs)
    if isDistributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)

    import argparse
    parser = argparse.ArgumentParser(
        description='ap film distributed training job')
    parser.add_argument('-cfg', '--config', dest="config",  type=str, default="config",
                        help='Configuration filename without extension. This file should be in the configs folder (default: config)')
    parser.add_argument('-ls', '--load-snapshot', dest="load_snapshot", action='store_true',
                        help='Load saved snapshot (default: False)')
    parser.add_argument('-dis', '--distributed', dest="distributed",
                        action='store_true', help='Distributed training (default: False)')
    args = parser.parse_args()

    main(args.config, args.distributed, args.load_snapshot)
