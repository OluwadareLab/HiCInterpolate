import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from torch import Tensor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


class PairDataset(Dataset):
    def __init__(self, pair_dicts: List):
        self.pair_dicts = pair_dicts

    def __len__(self):
        return len(self.pair_dicts)

    def get_image(self, image_file: str) -> Tensor:
        try:
            np_img = np.load(image_file)
            # Check for empty or zero-dimension arrays which cause 'stack' errors
            if np_img.size == 0 or 0 in np_img.shape:
                logging.warning(f"Corrupted or empty patch found: {image_file} with shape {np_img.shape}")
                return torch.empty(0)
            np_arr = np.ascontiguousarray(np_img, dtype=np.float32)
            img = torch.from_numpy(np_arr).unsqueeze(0)
            return img
        except Exception as e:
            logging.error(f"Failed to load patch file {image_file}: {e}")
            return torch.empty(0)

    def __getitem__(self, idx):
        key = self.pair_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        x1 = self.get_image(image_file=key["frame_1"])
        x2 = self.get_image(image_file=key["frame_2"])

        # Return None if any frame failed to load correctly
        if x0.numel() == 0 or x1.numel() == 0 or x2.numel() == 0:
            return None

        time = torch.tensor([key["time"]], dtype=torch.float32)

        return x0, x1, x2, time


class CustomDataset:
    def __init__(self, record_file: str, img_dir: str, img_map: dict):
        self.record_file = record_file
        self.img_dir = img_dir
        self.img_map = img_map

    def _prep_pairs(self):
        record_file = self.record_file
        with open(record_file, "r") as fid:
            pair_list = np.loadtxt(fid, dtype=str)

        image_dir = self.img_dir.rstrip(os.sep)
        base_path = image_dir
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
