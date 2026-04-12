import random
import numpy as np
import sys
import os
import logging
import torch
import torch.distributed as dist
import argparse
from src import TrainLib, TestLib, CustomDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
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


def get_dataloader(ds: Dataset, batch_size: int = 8, shuffle: bool = False,
                   isDistributed: bool = False, sampler=None) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(ds, shuffle=shuffle)
        )
    else:
        if sampler is not None:
            return DataLoader(
                ds,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=set_seed,
                sampler=sampler
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=set_seed
        )


def main(config_filename: str, isDistributed: bool = False, load_snapshot: bool = False, train: bool = False, test: bool = False):
    yaml_cfg = OmegaConf.load(f"./configs/{config_filename}.yaml")
    structured_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    # OmegaConf.update(cfg, "dir.root", "/home/mohit/Documents/project/interpolation/HiCInterpolate")
    # OmegaConf.update(cfg, "dir.data", "/home/mohit/Documents/project/interpolation/data/triplets/normalized")

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)
    OmegaConf.update(cfg, "dir.model_state", model_state_dir)

    log = base_logger(cfg.file.log)
    if isDistributed:
        ddp_setup()

    batch_size = cfg.data.batch_size
    preprocess_cfg = {
        "apply_log1p": bool(cfg.data.get("apply_log1p", False)),
        "signed_log1p": bool(cfg.data.get("signed_log1p", False)),
        "apply_normalization": bool(cfg.data.get("apply_normalization", False)),
        "normalization_mode": str(cfg.data.get("normalization_mode", "train_percentile")),
        "normalization_lower_percentile": float(cfg.data.get("normalization_lower_percentile", 0.1)),
        "normalization_upper_percentile": float(cfg.data.get("normalization_upper_percentile", 99.9)),
        "normalization_max_files": int(cfg.data.get("normalization_max_files", 0)),
        "normalization_sample_values_per_file": int(cfg.data.get("normalization_sample_values_per_file", 4096)),
        "normalization_fixed_low": float(cfg.data.get("normalization_fixed_low", 0.0)),
        "normalization_fixed_high": float(cfg.data.get("normalization_fixed_high", 1.0)),
        "norm_eps": float(cfg.data.get("norm_eps", 1e-8)),
    }

    cds = CustomDataset(record_file=cfg.file.dataset_dict, img_dir=cfg.dir.image,
                        img_map=cfg.data.interpolator_images_map,
                        shuffle=True,
                        train_val_test_ratio=cfg.data.train_val_test_ratio,
                        preprocess_cfg=preprocess_cfg)
    train_ds, val_ds, test_ds = cds._get_dataset()

    if train:
        train_sampler = None
        if cfg.data.sparse_sampling_enabled:
            if isDistributed:
                print("[WARN] sparse_sampling_enabled is ignored during distributed training.")
            else:
                sampling_weights = train_ds.build_sampling_weights(
                    nonzero_threshold=cfg.data.sparse_sampling_nonzero_threshold,
                    informative_ratio=cfg.data.sparse_sampling_informative_ratio,
                    informative_boost=cfg.data.sparse_sampling_informative_boost,
                )
                train_sampler = WeightedRandomSampler(
                    weights=sampling_weights,
                    num_samples=len(sampling_weights),
                    replacement=True,
                )

        train_dl = get_dataloader(
            ds=train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            isDistributed=isDistributed,
            sampler=train_sampler)
        val_dl = get_dataloader(ds=val_ds, batch_size=batch_size,
                                shuffle=False, isDistributed=isDistributed)
        trainer = TrainLib.Trainer(cfg=cfg, log=log, train_dl=train_dl, val_dl=val_dl,
                                   load_snapshot=load_snapshot, isDistributed=isDistributed)
        trainer.train(max_epochs=cfg.training.epochs)

    if test and os.path.exists(cfg.file.model):
        test_dl = get_dataloader(
            ds=test_ds, batch_size=batch_size, shuffle=False, isDistributed=isDistributed)
        tester = TestLib.Tester(
            cfg=cfg, log=log, model=cfg.file.model, test_dl=test_dl, isDistributed=isDistributed)
        tester.test()

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

    main(args.config, args.distributed, args.load_snapshot, args.train, args.test)
