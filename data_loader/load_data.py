from torch import Tensor
from typing import List, Tuple
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import torch
import sys
import os
import numpy as np
import config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TripletDataset(Dataset):
    def __init__(self, triplet_dicts: List):
        self.triplet_dicts = triplet_dicts

    def __len__(self):
        return len(self.triplet_dicts)

    def get_image(self, image_file: str) -> Tensor:
        img = torch.from_numpy(np.load(image_file)).unsqueeze(0)
        return img

    def pad_to_64x64(self, x):
        _, h, w = x.shape
        pad_h = max(0, 64 - h)
        pad_w = max(0, 64 - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def __getitem__(self, idx):
        key = self.triplet_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        y = self.get_image(image_file=key["frame_1"])
        x1 = self.get_image(image_file=key["frame_2"])
        time = torch.tensor([key["time"]], dtype=torch.float32)

        # x0 = self.pad_to_64x64(x0)
        # y = self.pad_to_64x64(y)
        # x1 = self.pad_to_64x64(x1)

        return x0, y, x1, time


def get_train_val_dl() -> Tuple[Dataset, Dataset]:
    record_file = config.RECORD_FILE
    with open(record_file, "r") as fid:
        triplets_list = np.loadtxt(fid, dtype=str)

    image_dir = config.IMAGE_DIR
    image_map = config.INTERPOLATOR_IMAGES_MAP
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

    train_len = int(config.TRAIN_VAL_RATIO[0] * num_of_sample)
    val_len = int(config.TRAIN_VAL_RATIO[1] * num_of_sample)
    train_dicts = triplet_dicts[:train_len]
    val_dicts = triplet_dicts[train_len:train_len+val_len]

    train_ds = TripletDataset(
        triplet_dicts=train_dicts)
    val_ds = TripletDataset(triplet_dicts=val_dicts)

    return train_ds, val_ds


if __name__ == '__main__':
    get_train_val_dl()
