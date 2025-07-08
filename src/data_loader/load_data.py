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
    def __init__(self, record_file: str, img_dir: str, img_map: dict, train_val_ratio: List = [0, 0]):
        self.record_file = record_file
        self.img_dir = img_dir
        self.img_map = img_map
        self.train_val_ratio = train_val_ratio

    def _prep_triplets(self):
        record_file = self.record_file
        with open(record_file, "r") as fid:
            triplets_list = np.loadtxt(fid, dtype=str)

        image_dir = self.img_dir
        image_map = self.img_map
        triplet_dicts = []
        for triplet in triplets_list:
            triplet_dict = {
                image_key: os.path.join(image_dir, triplet, image_basename)
                for image_key, image_basename in image_map.items()
            }
            triplet_dict["time"] = 0.5
            triplet_dicts.append(triplet_dict)

        random.shuffle(triplet_dicts)

        return triplet_dicts

    def _get_test_dl(self) -> Dataset:
        triplet_dicts = self._prep_triplets()
        test_ds = TripletDataset(triplet_dicts=triplet_dicts)
        return test_ds

    def _get_train_dl(self) -> Tuple[Dataset, Dataset]:
        triplet_dicts = self._prep_triplets()
        num_of_sample = len(triplet_dicts)
        train_len = int(self.train_val_ratio[0] * num_of_sample)
        val_len = int(self.train_val_ratio[1] * num_of_sample)
        train_dicts = triplet_dicts[:train_len]
        val_dicts = triplet_dicts[train_len:train_len+val_len]

        train_ds = TripletDataset(
            triplet_dicts=train_dicts)
        val_ds = TripletDataset(triplet_dicts=val_dicts)

        return train_ds, val_ds
