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
DICT_DIR = f'{ROOT_DIR}/datasets/timeseries/full_triplets/inference_full'
IMAGE_DIR = f'{ROOT_DIR}/datasets/timeseries/full_triplets'

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


def get_prediction(batch_size, dataset_dict, model: nn.Module, resol, patch, organism, chromosome):
    cds = CustomDataset(record_file=dataset_dict, img_dir=IMAGE_DIR,
                        img_map=IMAGE_MAP)
    dataset_dict = cds._get_dataset()
    dataset = TripletDataset(triplet_dicts=dataset_dict)
    test_dl = get_dataloader(ds=dataset, batch_size=batch_size)

    N = math.ceil(CHROMOSOME_SIZES[organism][chromosome]/resol)
    y_matrix = np.zeros((N, N))
    pred_matrix = np.zeros((N, N))

    pred_list = []
    y_list = []

    count = 0
    for _, batch in enumerate(tqdm(test_dl)):
        count += 1
        if batch is None:
            continue
        x1, y, x2, upper_x0, upper_y, upper_x1 = batch
        upper_x0 = upper_x0.to(DEVICE)
        upper_y = upper_y.to(DEVICE)
        upper_x1 = upper_x1.to(DEVICE)
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        y = y.to(DEVICE)
        y = y*upper_y

        pred = model(x1, x2)
        pred = pred*upper_y

        pred_list.append(pred.detach().cpu())
        y_list.append(y.detach().cpu())

        if pred.shape != y.shape:
            print(f"[WARNING] Shape mismatch..... {count} "
                  f"pred={pred.shape}, target={y.shape}")
        if pred.shape != (1, 1, patch, patch) or y.shape != (1, 1, patch, patch):
            print(f"[WARNING] Patch shape mismatch for ... {count} "
                  f"pred={pred.shape}, target={y.shape}")

        # del x1, y, x2, upper_x0, upper_y, upper_x1, pred

    if len(pred_list) == 0 or len(y_list) == 0:
        raise RuntimeError(f"No valid batches for {dataset_dict}")

    if len(pred_list) * len(pred_list) != len(pred_list) * len(y_list):
        print(f"[WARNING] Patch count mismatch for {dataset_dict}: "
              f"pred={len(pred_list)}, target={len(y_list)}")

    print(
        f"[WARNING] Reconstructing from {len(y_list)} patches with patch size {patch}")
    print(f"[WARNING] sqrt(total_patches) assumes square grid; this may fail if tiles are incomplete")

    y_tensor = torch.stack(y_list, dim=0)
    y_tensor = y_tensor.permute(1, 0, 2, 3, 4)
    y_matrix = reconstruct_from_model_outputs(y_tensor, patch)
    y_matrix = y_matrix[0][0][0:N, 0:N]
    y_matrix[y_matrix < 0] = 0.0

    predictions_tensor = torch.stack(pred_list, dim=0)
    predictions_tensor = predictions_tensor.permute(1, 0, 2, 3, 4)
    pred_matrix = reconstruct_from_model_outputs(predictions_tensor, patch)
    pred_matrix = pred_matrix[0][0][0:N, 0:N]
    pred_matrix[pred_matrix < 0] = 0.0

    print(f"Processed {count} batches")
    print(
        f"y_matrix shape: {y_matrix.shape}, pred_matrix shape: {pred_matrix.shape}")

    return y_matrix, pred_matrix


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
                                    record_filename = os.path.join(
                                        DICT_DIR, f'{resolution}_{model_patch}_{organism}_{sample}_{subsample}_{triplet[1]}_chr{chromosome}.inference_full')
                                    print(
                                        f"Running inference for record: {record_filename}")
                                    LOG.info(
                                        f"Running inference for record: {record_filename}")

                                    matrix_filename_prefix = f"{heatmap_output_dir}/{resolution}_{model_patch}_{organism}_{sample}_{subsample}_{triplet[1]}_{chromosome}"
                                    y_matrix, pred_matrix = get_prediction(batch_size=BATCH_SIZE, dataset_dict=record_filename, model=model,
                                                                           resol=resolution, patch=model_patch, organism=organism, chromosome=chromosome)

                                    np.save(
                                        f"{matrix_filename_prefix}_y.npy", y_matrix)
                                    np.save(
                                        f"{matrix_filename_prefix}_pred.npy", pred_matrix)


if __name__ == "__main__":
    run_inference()
