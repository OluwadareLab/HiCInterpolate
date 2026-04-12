import numpy as np
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from torch import Tensor
import sys
import os
from scipy.stats import norm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_EPSILON = 1e-8


_min = torch.tensor(_EPSILON).float()
_max = torch.tensor(1.0).float()


def _apply_log_transform(np_img: np.ndarray, apply_log1p: bool, signed_log1p: bool) -> np.ndarray:
    if not apply_log1p:
        return np_img

    if signed_log1p:
        return np.sign(np_img) * np.log1p(np.abs(np_img))

    # Standard log1p branch for non-negative count-like matrices.
    return np.log1p(np.clip(np_img, a_min=0.0, a_max=None))


def _apply_minmax_normalization(np_img: np.ndarray, norm_stats: Optional[Dict[str, float]], eps: float) -> np.ndarray:
    if norm_stats is None:
        return np_img

    lo = float(norm_stats["low"])
    hi = float(norm_stats["high"])
    np_img = np.clip(np_img, lo, hi)
    denom = max(hi - lo, eps)
    return (np_img - lo) / denom


class TripletDataset(Dataset):
    def __init__(self, triplet_dicts: List,
                 preprocess_cfg: Optional[Dict] = None,
                 normalization_stats: Optional[Dict[str, float]] = None):
        self.triplet_dicts = triplet_dicts
        self.preprocess_cfg = preprocess_cfg or {}
        self.normalization_stats = normalization_stats

    def __len__(self):
        return len(self.triplet_dicts)

    def get_image(self, image_file: str) -> Tensor:
        np_img = np.asarray(np.load(image_file), dtype=np.float32)
        apply_log1p = bool(self.preprocess_cfg.get("apply_log1p", False))
        signed_log1p = bool(self.preprocess_cfg.get("signed_log1p", False))
        apply_normalization = bool(
            self.preprocess_cfg.get("apply_normalization", False))
        norm_eps = float(self.preprocess_cfg.get("norm_eps", _EPSILON))

        np_img = _apply_log_transform(np_img, apply_log1p, signed_log1p)
        if apply_normalization:
            np_img = _apply_minmax_normalization(
                np_img, self.normalization_stats, norm_eps)
        # norm_img = norm.pdf(np_img)
        # mm_img = np.clip(norm_img, 0.0, 1.0)
        img = torch.from_numpy(np_img).float().unsqueeze(0)
        return img

    def __getitem__(self, idx):
        key = self.triplet_dicts[idx]
        x0 = self.get_image(image_file=key["frame_0"])
        y = self.get_image(image_file=key["frame_1"])
        x1 = self.get_image(image_file=key["frame_2"])
        time = torch.tensor([key["time"]], dtype=torch.float32)

        return x0, y, x1, time

    def build_sampling_weights(self, nonzero_threshold: float = 1e-6,
                               informative_ratio: float = 0.02,
                               informative_boost: float = 3.0) -> torch.Tensor:
        weights = []
        denom = max(informative_ratio, 1e-12)
        for key in self.triplet_dicts:
            target = np.load(key["frame_1"], mmap_mode="r")
            nonzero_ratio = float(np.mean(target > nonzero_threshold))
            weight = 1.0 + informative_boost * min(1.0, nonzero_ratio / denom)
            weights.append(weight)
        return torch.as_tensor(weights, dtype=torch.double)


class CustomDataset:
    def __init__(self, record_file: str, img_dir: str, img_map: dict, shuffle: bool = True,
                 train_val_test_ratio: List = [0.6, 0.2, 0.2], preprocess_cfg: Optional[Dict] = None):
        self.record_file = record_file
        self.img_dir = img_dir
        self.img_map = img_map
        self.train_val_test_ratio = train_val_test_ratio
        self.shuffle = shuffle
        self.preprocess_cfg = preprocess_cfg or {}

    def _compute_normalization_stats(self, train_dicts: List) -> Optional[Dict[str, float]]:
        if not bool(self.preprocess_cfg.get("apply_normalization", False)):
            return None

        mode = str(self.preprocess_cfg.get("normalization_mode", "train_percentile"))
        if mode == "fixed":
            return {
                "low": float(self.preprocess_cfg.get("normalization_fixed_low", 0.0)),
                "high": float(self.preprocess_cfg.get("normalization_fixed_high", 1.0)),
            }

        lower_p = float(self.preprocess_cfg.get("normalization_lower_percentile", 0.1))
        upper_p = float(self.preprocess_cfg.get("normalization_upper_percentile", 99.9))
        max_files = int(self.preprocess_cfg.get("normalization_max_files", 0))
        sample_values_per_file = int(
            self.preprocess_cfg.get("normalization_sample_values_per_file", 4096))
        apply_log1p = bool(self.preprocess_cfg.get("apply_log1p", False))
        signed_log1p = bool(self.preprocess_cfg.get("signed_log1p", False))
        norm_eps = float(self.preprocess_cfg.get("norm_eps", _EPSILON))

        selected_dicts = train_dicts if max_files <= 0 else train_dicts[:max_files]
        sampled_values = []
        for key in selected_dicts:
            target = np.asarray(np.load(key["frame_1"], mmap_mode="r"), dtype=np.float32)
            target = _apply_log_transform(target, apply_log1p, signed_log1p)
            flat = target.reshape(-1)
            if sample_values_per_file > 0 and flat.size > sample_values_per_file:
                idx = np.random.choice(flat.size, size=sample_values_per_file, replace=False)
                flat = flat[idx]
            sampled_values.append(flat)

        if len(sampled_values) == 0:
            return None

        values = np.concatenate(sampled_values)
        low = float(np.percentile(values, lower_p))
        high = float(np.percentile(values, upper_p))
        if high <= low + norm_eps:
            high = low + norm_eps
        return {"low": low, "high": high}

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

        normalization_stats = self._compute_normalization_stats(train_dicts)

        train_ds = TripletDataset(
            triplet_dicts=train_dicts,
            preprocess_cfg=self.preprocess_cfg,
            normalization_stats=normalization_stats)
        val_ds = TripletDataset(
            triplet_dicts=val_dicts,
            preprocess_cfg=self.preprocess_cfg,
            normalization_stats=normalization_stats)
        test_ds = TripletDataset(
            triplet_dicts=test_dicts,
            preprocess_cfg=self.preprocess_cfg,
            normalization_stats=normalization_stats)

        return train_ds, val_ds, test_ds
