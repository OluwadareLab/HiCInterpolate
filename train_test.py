import random
import numpy as np
import pandas as pd
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


def set_seed(seed: int = 42):
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def ddp_setup():
    print(f"[INFO] setting up distributed parallel environment...")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    print(f"[INFO] environment setting done!")


def prepare_dataloader(train_ds: Dataset, val_ds: Dataset, batch_size: int = 0, isDistributed: bool = False) -> Tuple[Dataset, Dataset]:
    if isDistributed:
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(train_ds)
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True
        )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )
    return train_dl, val_dl


def main(total_epochs: int, batch_size: int, save_every: int, load_snapshot: bool, augmentation: bool = False, isDistributed: bool = False):
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

    # import argparse
    # parser = argparse.ArgumentParser(
    #     description='ap film distributed training job')
    # parser.add_argument('-e', '--epochs', dest="epochs",  type=int, default=100,
    #                     help='Total epochs to train the model (default: 100)')
    # parser.add_argument('-se', '--save-every', dest="save_every", type=int, default=20,
    #                     help='How often to save a snapshot (default: 20)')
    # parser.add_argument('-bs', '--batch-size', dest="batch_size", type=int, default=8,
    #                     help='Input batch size on each device (default: 8)')
    # parser.add_argument('-ls', '--load-snapshot', dest="load_snapshot", action='store_true',
    #                     help='Load saved snapshot (default: False)')
    # parser.add_argument('-da', '--data-augmentation', dest="data_augmentation",
    #                     action='store_true', help='Train data augmentation (default: False)')
    # parser.add_argument('-dis', '--distributed', dest="distributed",
    #                     action='store_true', help='Distributed training (default: False)')
    # args = parser.parse_args()

    # main(args.epochs, args.batch_size, args.save_every,
    #      args.load_snapshot, args.data_augmentation, args.distributed)

    main(1000, 8, 5, False, True, False)
