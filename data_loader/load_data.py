import math
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch import Tensor
from typing import List, Tuple
from torch.utils.data import Dataset
import random
import torch
import sys
import os
import numpy as np
import config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TripletDataset(Dataset):
    def __init__(self, triplet_dicts: List, augment=False, crop=False):
        self.triplet_dicts = triplet_dicts
        self.augment = augment
        self.crop_f = transforms.CenterCrop(config.CROP_SIZE)
        self.crop = crop

    def __len__(self):
        return len(self.triplet_dicts)

    def get_image(self, image_file: str) -> Tensor:
        img = torch.from_numpy(np.load(image_file))
        return img

    def get_image_range(self, img, raw=False) -> Tensor:
        if raw:
            tensor_img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(img.tobytes()))
            tensor_img = tensor_img.view(
                img.size[1], img.size[0], 3).permute(2, 0, 1)
        else:
            to_tensor = transforms.ToTensor()
            tensor_img = to_tensor(img)
        return tensor_img

    def random_rot90(self, x0, y, x1):
        k = random.randint(0, 4)
        if k > 0:
            x0 = TF.rotate(x0, angle=90 * k)
            y = TF.rotate(y, angle=90 * k)
            x1 = TF.rotate(x1, angle=90 * k)
        return x0, y, x1

    def random_flip(self, x0, y, x1):
        if random.random() < 0.5:
            x0 = TF.hflip(x0)
            y = TF.hflip(y)
            x1 = TF.hflip(x1)
        return x0, y, x1

    def random_reverse(self, x0, y, x1):
        if random.random() < 0.7:
            x0, y = y, x0
        return x0, y, x1

    def random_rot(self, x0, y, x1):
        k = random.randint(0, 2)
        if k > 0:
            angle = random.uniform(-0.25 * math.pi, 0.25 * math.pi)
            x0 = TF.rotate(x0, angle=angle,
                           interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            y = TF.rotate(y, angle=angle,
                          interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            x1 = TF.rotate(x1, angle=angle,
                           interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        return x0, y, x1

    def apply_same_augmentation(self, x0, y, x1):
        x0, y, x1 = self.random_rot90(x0, y, x1)
        x0, y, x1 = self.random_flip(x0, y, x1)
        x0, y, x1 = self.random_reverse(x0, y, x1)
        x0, y, x1 = self.random_rot(x0, y, x1)

        return x0, y, x1

    def __getitem__(self, idx):
        key = self.triplet_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        y = self.get_image(image_file=key["frame_1"])
        x1 = self.get_image(image_file=key["frame_2"])
        time = torch.tensor([key["time"]], dtype=torch.float32)

        if self.augment:
            x0, y, x1 = self.apply_same_augmentation(x0, y, x1)

        x0 = self.get_image_range(x0)
        y = self.get_image_range(y)
        x1 = self.get_image_range(x1)

        if self.crop:
            x0 = self.crop_f(x0)
            y = self.crop_f(y)
            x1 = self.crop_f(x1)

        return x0, y, x1, time


def get_train_val_dl(augmentation: bool = False) -> Tuple[Dataset, Dataset]:
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
        triplet_dicts=train_dicts, augment=augmentation, crop=True)
    val_ds = TripletDataset(triplet_dicts=val_dicts, augment=False, crop=True)

    return train_ds, val_ds


if __name__ == '__main__':
    get_train_val_dl()
