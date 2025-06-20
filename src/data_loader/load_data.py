import numpy as np
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple
from torch import Tensor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TripletDataset(Dataset):
    def __init__(self, triplet_dicts: List):
        self.triplet_dicts = triplet_dicts

    def __len__(self):
        return len(self.triplet_dicts)

    def get_image(self, image_file: str) -> Tensor:
        img = torch.from_numpy(np.load(image_file)).unsqueeze(0)
        return img

    def __getitem__(self, idx):
        key = self.triplet_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        y = self.get_image(image_file=key["frame_1"])
        x1 = self.get_image(image_file=key["frame_2"])
        time = torch.tensor([key["time"]], dtype=torch.float32)

        return x0, y, x1, time


class CustomDataset:
    def __init__(self, config):
        self.cfg = config

    def get_train_val_dl(self) -> Tuple[Dataset, Dataset]:
        record_file = self.cfg.paths.record_file
        with open(record_file, "r") as fid:
            triplets_list = np.loadtxt(fid, dtype=str)

        image_dir = self.cfg.paths.image_dir
        image_map = self.cfg.data.interpolator_images_map
        triplet_dicts = []
        for triplet in triplets_list:
            triplet_dict = {
                image_key: os.path.join(image_dir, triplet, image_basename)
                for image_key, image_basename in image_map.items()
            }
            triplet_dict["time"] = 0.5
            triplet_dicts.append(triplet_dict)
        num_of_sample = len(triplet_dicts)
        random.shuffle(triplet_dicts)

        train_len = int(self.cfg.data.train_val_ratio[0] * num_of_sample)
        val_len = int(self.cfg.data.train_val_ratio[1] * num_of_sample)
        train_dicts = triplet_dicts[:train_len]
        val_dicts = triplet_dicts[train_len:train_len+val_len]

        train_ds = TripletDataset(
            triplet_dicts=train_dicts)
        val_ds = TripletDataset(triplet_dicts=val_dicts)

        return train_ds, val_ds
