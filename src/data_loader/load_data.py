import numpy as np
from numpy.random import sample
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple
from torch import Tensor
import sys
import os
from scipy.stats import norm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


class TripletDataset(Dataset):
    def __init__(self, triplet_dicts: List):
        self.triplet_dicts = triplet_dicts

    def __len__(self):
        return len(self.triplet_dicts)

    def get_image(self, image_file: str) -> Tensor:
        img = np.load(image_file)
        # Filter out matrices with NaNs, Infs, or those that are "identical" (constant values like all zeros)
        if np.isnan(img).any() or np.isinf(img).any() or img.min() == img.max():
            return None
        # img = self.log1p(img)
        # img = self.min_max_norm(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img

    def log1p(self, matrix):
        return np.log1p(matrix)

    def min_max_norm(self, matrix):
        _min = matrix.min()
        _max = matrix.max()
        denominator = _max - _min

        if denominator > 0:
            mm_mat = (matrix - _min) / denominator
        else:
            mm_mat = matrix - _min

        return mm_mat

    def __getitem__(self, idx):
        key = self.triplet_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        y = self.get_image(image_file=key["frame_1"])
        x1 = self.get_image(image_file=key["frame_2"])
        time = torch.tensor([key["time"]], dtype=torch.float32)
        if x0 is None or y is None or x1 is None:
            return None

        return x0, y, x1, time


class CustomDataset:
    def __init__(self, record_file: str, img_dir: str, img_map: dict, shuffle: bool = True, train_val_test_ratio: List = [0.6, 0.2, 0.2]):
        self.record_file = record_file
        self.img_dir = img_dir
        self.img_map = img_map
        self.train_val_test_ratio = train_val_test_ratio
        self.shuffle = shuffle

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

        if self.shuffle:
            random.shuffle(triplet_dicts)

        return triplet_dicts

    def _get_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        triplet_dicts = self._prep_triplets()
        num_of_sample = len(triplet_dicts)

        train_len = int(self.train_val_test_ratio[0] * num_of_sample)
        val_len = int(self.train_val_test_ratio[1] * num_of_sample)
        test_len = int(self.train_val_test_ratio[2] * num_of_sample)

        train_dicts = triplet_dicts[:train_len]
        val_dicts = triplet_dicts[train_len:train_len+val_len]
        test_dicts = triplet_dicts[train_len +
                                   val_len:train_len+val_len+test_len]

        train_ds = TripletDataset(
            triplet_dicts=train_dicts)
        val_ds = TripletDataset(triplet_dicts=val_dicts)
        test_ds = TripletDataset(triplet_dicts=test_dicts)

        return train_ds, val_ds, test_ds
