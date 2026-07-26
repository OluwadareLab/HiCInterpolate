import math
from src.interpolator.model import Interpolator
from src.data_loader.inference_data_loader import CustomDataset, TripletDataset
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import os
import random
import logging
import numpy as np
import torch
from flow_based_interpolation import of_interpolation as OF
from _4DMax import model as _4DMax_model

ROOT_DIR = '/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate'
MODEL_DIR = f'{ROOT_DIR}/datasets/final_output/triplets_dataset'
IMAGE_DIR = f'{ROOT_DIR}/datasets/timeseries/full_matrix_triplets'

OUTPUT_DIR = f'{ROOT_DIR}/datasets/timeseries/full_triplets/output'
LOG_FILE = f'{OUTPUT_DIR}/inference_full.log'

BATCH_SIZE = 1

RESOLUTIONS = [25000]
PATCHES = [64]
MODEL_BATCHES = [20]

CHROMOSOMES = {
    "human": ["10", "11", "15", "16", "20", "21"],
    "mouse": ["10", "15", "19"]
}

CHROMOSOME_SIZES = {
    "human": {
        "10": 133797422,
        "11": 135086622,
        "15": 101991189,
        "16": 90338345,
        "20": 64444167,
        "21": 46709983
    },
    "mouse": {
        "10": 130694993,
        "15": 104043685,
        "19": 61431566
    }
}

TEST_DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                    [
                        ["dmso_control_30m",
                         "dmso_control_60m",
                         "dmso_control_90m"]
                    ]
            }
        },
        "dtag": {
            "v1": {
                "triplets":
                    [
                        ["dtag_v1_30m",
                         "dtag_v1_60m",
                         "dtag_v1_90m"]
                    ]
            }
        }
    },
    "mouse": {
        "cerebellar_granule_neuron": {
            "control": {
                "triplets":
                [
                    ["cerebellar_granule_neuron_control_10080m",
                     "cerebellar_granule_neuron_control_11520m",
                     "cerebellar_granule_neuron_control_12960m"]
                ]
            }
        },
        "embryo": {
            "development": {
                "triplets": [
                    ["zygote",
                     "early2_cell",
                     "late2_cell"],

                    ["early2_cell",
                     "late2_cell",
                     "8cell"],

                    ["late2_cell",
                     "8cell",
                     "icm"]
                ]
            }
        }
    }
}

IMAGE_MAP = {
    'frame_0': 'img1.npy',
    'frame_1': 'img2.npy',
    'frame_2': 'img3.npy'
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG = base_logger(LOG_FILE)


def get_res_tag(resolution: int) -> str:
    if resolution == 25000:
        return "25k"
    elif resolution == 10000:
        return "10k"
    elif resolution == 5000:
        return "5k"
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def set_seed(seed_v: int = 42):
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def get_dataloader(ds: Dataset, batch_size: int = 1) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        shuffle=False,
        worker_init_fn=set_seed,
        num_workers=4,
        persistent_workers=True
    )


def _remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict


@torch.no_grad()
def load_model(model_path: str) -> nn.Module:
    model = Interpolator().to(DEVICE)
    snapshot = torch.load(model_path, map_location=DEVICE)
    state_dict = _remove_module_prefix(snapshot['model'])
    model.load_state_dict(state_dict)

    model.eval()
    return model


def reconstruct_from_model_outputs(predictions, P):
    B, total_patches, C, _, _ = predictions.shape
    num_patches_per_dim = int(math.sqrt(total_patches))
    N = num_patches_per_dim * P
    out = predictions.view(B, num_patches_per_dim,
                           num_patches_per_dim, C, P, P)
    out = out.permute(0, 3, 1, 4, 2, 5)
    reconstructed_matrix = out.reshape(B, C, N, N)
    return reconstructed_matrix


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return (1.0 - t) * x0 + t * x1


def get_prediction(batch_size, dataset_dir, model: nn.Module, resol, patch, organism, chromosome):
    N = math.ceil(CHROMOSOME_SIZES[organism][chromosome]/resol)

    x1 = np.load(os.path.join(dataset_dir, 'img1.npy'))
    x2 = np.load(os.path.join(dataset_dir, 'img3.npy'))
    x1 = torch.from_numpy(x1).to(DEVICE, dtype=torch.float32)
    x2 = torch.from_numpy(x2).to(DEVICE, dtype=torch.float32)

    x1 = x1.unsqueeze(0).unsqueeze(0)
    x2 = x2.unsqueeze(0).unsqueeze(0)

    print(f"[INFO] Running inference with Linear interpolation")
    LOG.info(f"[INFO] Running inference with Linear interpolation")
    linear = linear_interpolation(x1, x2, t=0.5)
    linear = linear.squeeze().squeeze().detach().cpu().numpy()
    linear = linear[0:N, 0:N]

    print(f"[INFO] Running inference with Optical Flow interpolation")
    LOG.info(f"[INFO] Running inference with Optical Flow interpolation")
    of = OF(x1, x2)
    of = of.squeeze().squeeze().detach().cpu().numpy()
    of = of[0:N, 0:N]

    print(f"[INFO] Running inference with 4DMax interpolation")
    LOG.info(f"[INFO] Running inference with 4DMax interpolation")
    _4dmax = _4DMax_model.run_4dmax(timeframe=[x1, x2], patch_size=patch)
    _4dmax = _4dmax.squeeze().squeeze().detach().cpu().numpy()

    h, w = _4dmax.shape
    if h < N or w < N:
        print(f"[WARNING] Padding 4DMax output from ({h}, {w}) to ({N}, {N})")
        LOG.info(
            f"[WARNING] Padding 4DMax output from ({h}, {w}) to ({N}, {N})")
        pad_h = N - h if h < N else 0
        pad_w = N - w if w < N else 0
        _4dmax = np.pad(
            _4dmax, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    elif h > N or w > N:
        _4dmax = _4dmax[0:N, 0:N]

    return linear, of, _4dmax


def run_inference():
    for model_res in RESOLUTIONS:
        for model_patch, model_batch in zip(PATCHES, MODEL_BATCHES):
            config_name = f'config_{get_res_tag(model_res)}_{model_patch}'

            model_output_dir = os.path.join(OUTPUT_DIR, config_name)
            os.makedirs(model_output_dir, exist_ok=True)

            heatmap_output_dir = os.path.join(
                model_output_dir)
            os.makedirs(heatmap_output_dir, exist_ok=True)

            model_filename = '/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/output/hicinterpolate/config_25k_64/hicinterpolate_64.pt'
            model = load_model(model_filename)
            print(f"Running inference for model: {model_filename}")
            LOG.info(f"Running inference for model: {model_filename}")

            for resolution in RESOLUTIONS:
                for organism, samples in TEST_DATASET.items():
                    for sample, condition in samples.items():
                        for subsample, content in condition.items():
                            for triplet in content["triplets"]:
                                for chromosome in CHROMOSOMES[organism]:
                                    record_path = os.path.join(
                                        IMAGE_DIR, f"{resolution}/{organism}/{sample}/{subsample}/{triplet[1]}/chr{chromosome}")

                                    print(
                                        f"Running inference for record: {record_path}")
                                    LOG.info(
                                        f"Running inference for record: {record_path}")

                                    matrix_filename_prefix = f"{heatmap_output_dir}/{resolution}_{model_patch}_{organism}_{sample}_{subsample}_{triplet[1]}_{chromosome}"
                                    linear, of, _4dmax = get_prediction(batch_size=BATCH_SIZE, dataset_dir=record_path, model=model,
                                                                        resol=resolution, patch=model_patch, organism=organism, chromosome=chromosome)

                                    print(
                                        f"Saving results for record: {matrix_filename_prefix}")
                                    LOG.info(
                                        f"Saving results for record: {matrix_filename_prefix}")
                                    np.save(
                                        f"{matrix_filename_prefix}_linear.npy", linear)
                                    np.save(
                                        f"{matrix_filename_prefix}_of.npy", of)
                                    np.save(
                                        f"{matrix_filename_prefix}_4dmax.npy", _4dmax)


if __name__ == "__main__":
    run_inference()
