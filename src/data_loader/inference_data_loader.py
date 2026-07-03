import numpy as np
from numpy.random import sample
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple
from torch import Tensor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


class TripletDataset(Dataset):
    def __init__(self, triplet_dicts: List):
        self.triplet_dicts = triplet_dicts

    def __len__(self):
        return len(self.triplet_dicts)

    def _load_valid_matrix(self, image_file: str):
        img = np.load(image_file)
        if np.isnan(img).any() or np.isinf(img).any() or img.min() == img.max():
            return None
        return img

    def log1p(self, matrix):
        return np.log1p(matrix)

    def normalize_triplet(self, x0, y, x1):
        logged = [self.log1p(matrix) for matrix in (x0, y, x1)]

        upper = max(
            np.percentile(matrix, CLIPPING_PERCENTILE)
            for matrix in logged
        )

        if upper <= _EPSILON:
            return None

        images = torch.stack([
            torch.from_numpy(
                (np.clip(matrix, 0.0, upper) / upper).astype(np.float32)
            )
            for matrix in logged
        ], dim=0)

        max_values = torch.tensor(
            [np.max(x0), np.max(y), np.max(x1)],
            dtype=torch.float32
        )

        packed = torch.cat([
            images.flatten(),
            max_values
        ])

        return packed

    def __getitem__(self, idx):
        key = self.triplet_dicts[idx]
        x0_raw = self._load_valid_matrix(image_file=key["frame_0"])
        y_raw = self._load_valid_matrix(image_file=key["frame_1"])
        x1_raw = self._load_valid_matrix(image_file=key["frame_2"])
        if x0_raw is None or y_raw is None or x1_raw is None:
            return None

        normalized = self.normalize_triplet(x0_raw, y_raw, x1_raw)
        if normalized is None:
            return None
        H, W = x0_raw.shape
        images = normalized[:-3].reshape(3, H, W)
        upper_x0, upper_y, upper_x1 = normalized[-3:]

        upper_x0 = upper_x0.unsqueeze(0)
        upper_y = upper_y.unsqueeze(0)
        upper_x1 = upper_x1.unsqueeze(0)
        x0_tensor = images[0].unsqueeze(0)
        y_tensor = images[1].unsqueeze(0)
        x1_tensor = images[2].unsqueeze(0)
        return x0_tensor, y_tensor, x1_tensor, upper_x0, upper_y, upper_x1


class CustomDataset:
    def __init__(self, record_file: str, img_dir: str, img_map: dict):
        self.record_file = record_file
        self.img_dir = img_dir
        self.img_map = img_map

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
            triplet_dicts.append(triplet_dict)

        return triplet_dicts

    def _get_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        triplet_dicts = self._prep_triplets()

        return triplet_dicts
