import random
import numpy as np
import sys
import os
from plots import *
from metrics import *
from data_loader import load_data
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple
import train_lib as Train

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


def main(total_epochs: int, batch_size: int, save_every: int, augmentation: bool = False, isDistributed: bool = False, load_snapshot: bool = False):
    os.makedirs(f"{config.ROOT_DIR}/output", exist_ok=True)
    os.makedirs(f"{config.ROOT_DIR}/models", exist_ok=True)

    if isDistributed:
        ddp_setup()
    train_ds, val_ds = load_data.get_train_val_dl(augmentation=augmentation)
    train_dl, val_dl = prepare_dataloader(
        train_ds, val_ds, batch_size, isDistributed)
    trainer = Train.Trainer(train_dl, val_dl, save_every,
                            load_snapshot, isDistributed)
    trainer.train(total_epochs)
    if isDistributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)

    import argparse
    parser = argparse.ArgumentParser(
        description='ap film distributed training job')
    parser.add_argument('-e', '--epochs', dest="epochs",  type=int, default=100,
                        help='Total epochs to train the model (default: 100)')
    parser.add_argument('-se', '--save-every', dest="save_every", type=int, default=20,
                        help='How often to save a snapshot (default: 20)')
    parser.add_argument('-bs', '--batch-size', dest="batch_size", type=int, default=8,
                        help='Input batch size on each device (default: 8)')
    parser.add_argument('-ls', '--load-snapshot', dest="load_snapshot", action='store_true',
                        help='Load saved snapshot (default: False)')
    parser.add_argument('-da', '--data-augmentation', dest="data_augmentation",
                        action='store_true', help='Train data augmentation (default: False)')
    parser.add_argument('-dis', '--distributed', dest="distributed",
                        action='store_true', help='Distributed training (default: False)')
    args = parser.parse_args()

    main(args.epochs, args.batch_size, args.save_every,
         args.data_augmentation, args.distributed, args.load_snapshot)
