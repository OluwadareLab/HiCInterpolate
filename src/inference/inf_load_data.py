import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from torch import Tensor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PairDataset(Dataset):
    def __init__(self, pair_dicts: List):
        self.pair_dicts = pair_dicts

    def __len__(self):
        return len(self.pair_dicts)

    def get_image(self, image_file: str) -> Tensor:
        img = torch.from_numpy(np.load(image_file)).unsqueeze(0)
        return img

    def __getitem__(self, idx):
        key = self.pair_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        x1 = self.get_image(image_file=key["frame_2"])
        time = torch.tensor([key["time"]], dtype=torch.float32)

        return x0, x1, time


class CustomDataset:
    def __init__(self, record_file: str, img_dir: str, img_map: dict):
        self.record_file = record_file
        self.img_dir = img_dir
        self.img_map = img_map

    def _prep_pairs(self):
        record_file = self.record_file
        with open(record_file, "r") as fid:
            pair_list = np.loadtxt(fid, dtype=str)

        image_dir = self.img_dir
        image_dirs = image_dir.split(os.sep)
        base_path = os.sep.join(image_dirs[:9])
        image_map = self.img_map
        pair_dicts = []
        for pair in pair_list:
            pair_dict = {
                image_key: os.path.join(base_path, pair, image_basename)
                for image_key, image_basename in image_map.items()
            }
            pair_dict["time"] = 0.5
            pair_dicts.append(pair_dict)

        return pair_dicts

    def _get_inference_dl(self) -> Dataset:
        pair_dicts = self._prep_pairs()
        test_ds = PairDataset(pair_dicts=pair_dicts)
        return test_ds
