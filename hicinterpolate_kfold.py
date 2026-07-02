import random
import numpy as np
import sys
import os
import logging
from sklearn.model_selection import KFold
import torch
import torch.distributed as dist
import argparse
import cupy as cp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from src import CustomDatasetKF, TrainLib
from omegaconf import OmegaConf
from configs.config import Config
from src.data_loader.load_data_kfold import TripletDatasetKF
from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if not hasattr(cp, 'float32'):
    cp.float32 = np.float32
    cp.float64 = np.float64
    cp.int32 = np.int32
    cp.int64 = np.int64


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


def set_seed(seed_v: int = 42):
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def get_dataloader(ds: Dataset, batch_size: int = 20, shuffle: bool = False, isDistributed: bool = False) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=set_seed,
            num_workers=4,
            persistent_workers=True,
            sampler=DistributedSampler(ds, shuffle=shuffle)
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=set_seed,
            num_workers=4,
            persistent_workers=True
        )


def ddp_setup():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training requires CUDA/NCCL, but CUDA is not available.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Distributed training requires torchrun. Example: "
            "torchrun --standalone --nproc_per_node=<num_gpus> hicinterpolate.py --distributed --train --config <config>"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def main(config_filename: str, isDistributed: bool = False, load_snapshot: bool = False, train: bool = False, test: bool = False):
    yaml_cfg = OmegaConf.load(f"./configs/{config_filename}.yaml")
    structured_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    isDistributed = isDistributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if isDistributed:
        local_rank = ddp_setup()
        OmegaConf.update(cfg, "device", f"cuda:{local_rank}", force_add=True)

    # OmegaConf.update(cfg, "dir.root", "/home/mohit/Documents/project/interpolation/HiCInterpolate")
    # OmegaConf.update(cfg, "dir.data", "/home/mohit/Documents/project/interpolation/data/triplets/normalized")

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)
    OmegaConf.update(cfg, "dir.model_state", model_state_dir)

    log = base_logger(cfg.file.log)

    train_cds = CustomDatasetKF(record_file=f'{cfg.file.dataset_dict}.train', img_dir=cfg.dir.image,
                                img_map=cfg.data.interpolator_images_map, shuffle=True)
    train_dict = train_cds._get_dataset()
    val_cds = CustomDatasetKF(record_file=f'{cfg.file.dataset_dict}.val', img_dir=cfg.dir.image,
                              img_map=cfg.data.interpolator_images_map, shuffle=True)
    val_dict = val_cds._get_dataset()

    data_dict = train_dict + val_dict

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    batch_size = cfg.data.batch_size
    output_dir = f"{cfg.dir.output}"
    model_state_dir = f"{cfg.dir.model_state}"
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_dict)):
        print(f"--- FOLD {fold} ---")
        log.info(f"--- FOLD {fold} ---")

        sub_output_dir = f"{output_dir}/fold_{fold}"
        sub_model_state_dir = f"{model_state_dir}/fold_{fold}"
        os.makedirs(f"{sub_output_dir}", exist_ok=True)
        os.makedirs(f"{sub_model_state_dir}", exist_ok=True)
        OmegaConf.update(cfg, "dir.output", sub_output_dir)
        OmegaConf.update(cfg, "dir.model_state", sub_model_state_dir)

        train_set = [data_dict[i] for i in train_idx]
        train_ds = TripletDatasetKF(
            triplet_dicts=train_set)

        val_set = [data_dict[i] for i in val_idx]
        val_ds = TripletDatasetKF(triplet_dicts=val_set)

        train_dl = get_dataloader(
            ds=train_ds, batch_size=batch_size, shuffle=True, isDistributed=isDistributed)
        val_dl = get_dataloader(ds=val_ds, batch_size=batch_size,
                                shuffle=False, isDistributed=isDistributed)

        trainer = TrainLib.Trainer(cfg=cfg, log=log, train_dl=train_dl, val_dl=val_dl,
                                   load_snapshot=load_snapshot, isDistributed=isDistributed)
        trainer.train(max_epochs=cfg.training.epochs)

    if isDistributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser(
        description='ap film distributed training job')
    parser.add_argument('-cfg', '--config', dest="config",  type=str, default="config",
                        help='Configuration filename without extension. This file should be in the configs folder (default: config)')
    parser.add_argument('-ls', '--load-snapshot', dest="load_snapshot", action='store_true',
                        help='Load saved snapshot (default: False)')
    parser.add_argument('-dis', '--distributed', dest="distributed",
                        action='store_true', help='Distributed training (default: False)')
    parser.add_argument('-train', '--train', dest="train",
                        action='store_true', help='Train Model (default: False)')
    parser.add_argument('-test', '--test', dest="test",
                        action='store_true', help='Test Model (default: False)')
    args = parser.parse_args()

    # args.config = "config_25k_64_kfold"
    # args.train = True
    # args.test = False
    main(args.config, args.distributed, args.load_snapshot, args.train, args.test)


# torchrun --standalone --nproc_per_node=1 hicinterpolate.py --test --config config_a1_5k_p64_b128
