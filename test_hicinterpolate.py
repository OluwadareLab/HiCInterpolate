from time import time
from src.interpolator.model import Interpolator
import matplotlib.colors as mcolors
from src.metric import metrics as eval_metric
from src.data_loader.load_data import CustomDataset
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import os
import random
from src.misc import plots as plot
from flow_based_interpolation import of_interpolation as OF
import logging
from _4DMax import model as _4DMax_model

ROOT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate'
MODEL_DIR = f'{ROOT_DIR}/datasets/final_output/triplets_dataset'
DICT_DIR = f'{ROOT_DIR}/datasets/triplets_dataset/test'
IMAGE_DIR = f'{ROOT_DIR}/datasets/triplets_dataset'
OUTPUT_DIR = f'{ROOT_DIR}/datasets/final_output/Plots_HiCInterpolate/'
LOG_FILE = f'{ROOT_DIR}/datasets/final_output/Plots_HiCInterpolate/inference.log'
CSV_FILENAME_SUFFIX = "comparative_scores.csv"

METHODS = ("Hicinterpolate", "4DMax", "Linear", "Optical Flow")
METRICS = ("psnr", "ssim",  "spearman", 'scc',
           "genome_disco", "genome_disco2", "hicrep")
METRIC_PRECISION = 4

BATCH_SIZE = 1

RESOLUTIONS = [25000]
PATCHES = [64]
MODEL_BATCHES = [20]

CHROMOSOMES = {
    "human": ["10", "11", "15", "16", "20", "21"]
}

TEST_DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    ["4DNFI7T93SHL_dmso_control_30m",
                     "4DNFICF2Z2TG_dmso_control_60m",
                     "4DNFILL624WG_dmso_control_90m"]
                ]
            }
        },
        "dtag": {
            "v1": {
                "triplets":
                [
                    ["4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m"]
                ]
            }
        },
        "wtc11": {
            "atrial": {
                "triplets":
                    [
                        ["wtc11_atrial_2880m",
                         "wtc11_atrial_5760m",
                         "wtc11_atrial_8640m"]
                    ]
            },
            "ventricular": {
                "triplets":
                    [
                        ["wtc11_ventricular_2880m",
                         "wtc11_ventricular_5760m",
                         "wtc11_ventricular_8640m"]
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

CMAP_JUICEBOX = mcolors.LinearSegmentedColormap.from_list(
    "juicebox", ["#FFFFFF", "#FFAAAA", "#FF5555", "#FF0000", "#B30000"], N=256
)


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


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


def get_dataloader(ds: Dataset, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        shuffle=shuffle,
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


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return (1.0 - t) * x0 + t * x1


def get_prediction(batch_size, dataset_dict, model: nn.Module, heatmap_filename, resol, patch):
    cds = CustomDataset(record_file=dataset_dict, img_dir=IMAGE_DIR,
                        img_map=IMAGE_MAP, shuffle=True, train_val_test_ratio=[0.0, 0.0, 1.0])
    _, _, test_ds = cds._get_dataset()

    test_dl = get_dataloader(ds=test_ds, batch_size=batch_size, shuffle=False)
    
    psnr_list = []
    ssim_list = []
    spearman_list = []
    scc_list = []
    genome_disco_list = []
    genome_disco2_list = []
    hicrep_list = []

    _4dmax_psnr_list = []
    _4dmax_ssim_list = []
    _4dmax_spearman_list = []
    _4dmax_scc_list = []
    _4dmax_genome_disco_list = []
    _4dmax_genome_disco2_list = []
    _4dmax_hicrep_list = []

    of_psnr_list = []
    of_ssim_list = []
    of_spearman_list = []
    of_scc_list = []
    of_genome_disco_list = []
    of_genome_disco2_list = []
    of_hicrep_list = []

    linear_psnr_list = []
    linear_ssim_list = []
    linear_spearman_list = []
    linear_scc_list = []
    linear_genome_disco_list = []
    linear_genome_disco2_list = []
    linear_hicrep_list = []

    count = 0
    for _, batch in enumerate(tqdm(test_dl)):
        count += 1
        if batch is None:
            continue
        x1, target, x2 = batch
        x1 = x1.to(DEVICE)
        target = target.to(DEVICE)
        x2 = x2.to(DEVICE)
        pred = model(x1, x2)
        pred[pred < 0] = 0
        _4dmax_pred = _4DMax_model.run_4dmax(x1, x2)
        linear = linear_interpolation(x1, x2, t=0.5)
        of = OF(x1, x2)

        num_examples = min(1, len(target))

        plot.plot_hic_heatmap(target=target[:num_examples],
                              title="Ground Truth", filename_prefix=heatmap_filename, count=count)

        plot.plot_hic_heatmap(target=pred[:num_examples],
                              title="Ours", filename_prefix=heatmap_filename, count=count)
        psnr_list.append(eval_metric.get_psnr_from_tensor(pred, target).item())
        ssim_list.append(
            eval_metric.get_ms_ssim_from_tensor(pred, target).item())
        spearman_list.append(
            eval_metric.get_spearman_from_tensor(pred, target).item())
        scc_list.append(eval_metric.get_scc_from_tensor(pred, target).item())
        genome_disco_list.append(
            eval_metric.get_genome_disco_from_tensor(pred, target).item())
        genome_disco2_list.append(
            eval_metric.get_genome_disco2_from_tensor(pred, target).item())
        hicrep_list.append(
            eval_metric.get_hicrep_from_tensor(pred, target).item())

        if not torch.isnan(_4dmax_pred).any():
            plot.plot_hic_heatmap(target=_4dmax_pred[:num_examples],
                                  title="4DMax", filename_prefix=heatmap_filename, count=count)
            _4dmax_psnr_list.append(
                eval_metric.get_psnr_from_tensor(_4dmax_pred, target).item())
            _4dmax_ssim_list.append(
                eval_metric.get_ms_ssim_from_tensor(_4dmax_pred, target).item())
            _4dmax_spearman_list.append(
                eval_metric.get_spearman_from_tensor(_4dmax_pred, target).item())
            _4dmax_scc_list.append(
                eval_metric.get_scc_from_tensor(_4dmax_pred, target).item())
            _4dmax_genome_disco_list.append(
                eval_metric.get_genome_disco_from_tensor(_4dmax_pred, target).item())
            _4dmax_genome_disco2_list.append(
                eval_metric.get_genome_disco2_from_tensor(_4dmax_pred, target).item())
            _4dmax_hicrep_list.append(
                eval_metric.get_hicrep_from_tensor(_4dmax_pred, target).item())

        plot.plot_hic_heatmap(target=of[:num_examples],
                              title="Optical Flow", filename_prefix=heatmap_filename, count=count)
        of_psnr_list.append(
            eval_metric.get_psnr_from_tensor(of, target).item())
        of_ssim_list.append(
            eval_metric.get_ms_ssim_from_tensor(of, target).item())
        of_spearman_list.append(
            eval_metric.get_spearman_from_tensor(of, target).item())
        of_scc_list.append(eval_metric.get_scc_from_tensor(of, target).item())
        of_genome_disco_list.append(
            eval_metric.get_genome_disco2_from_tensor(of, target).item())
        of_genome_disco2_list.append(
            eval_metric.get_genome_disco2_from_tensor(of, target).item())
        of_hicrep_list.append(
            eval_metric.get_hicrep_from_tensor(of, target).item())

        plot.plot_hic_heatmap(target=linear[:num_examples],
                              title="Linear", filename_prefix=heatmap_filename, count=count)
        linear_psnr_list.append(
            eval_metric.get_psnr_from_tensor(linear, target).item())
        linear_ssim_list.append(
            eval_metric.get_ms_ssim_from_tensor(linear, target).item())
        linear_spearman_list.append(
            eval_metric.get_spearman_from_tensor(linear, target).item())
        linear_scc_list.append(
            eval_metric.get_scc_from_tensor(linear, target).item())
        linear_genome_disco_list.append(
            eval_metric.get_genome_disco_from_tensor(linear, target).item())
        linear_genome_disco2_list.append(
            eval_metric.get_genome_disco2_from_tensor(linear, target).item())
        linear_hicrep_list.append(
            eval_metric.get_hicrep_from_tensor(linear, target).item())

        del x1, target, x2, pred, _4dmax_pred, of, linear

    metrics = {
        "psnr": {
            'ours': np.nanmean(psnr_list),
            '4dmax': np.nanmean(_4dmax_psnr_list),
            'optical_flow': np.nanmean(of_psnr_list),
            'linear': np.nanmean(linear_psnr_list)
        },
        "ssim": {
            'ours': np.nanmean(ssim_list),
            '4dmax': np.nanmean(_4dmax_ssim_list),
            'optical_flow': np.nanmean(of_ssim_list),
            'linear': np.nanmean(linear_ssim_list)
        },
        "spearman": {
            'ours': np.nanmean(spearman_list),
            '4dmax': np.nanmean(_4dmax_spearman_list),
            'optical_flow': np.nanmean(of_spearman_list),
            'linear': np.nanmean(linear_spearman_list)
        },
        "scc": {
            'ours': np.nanmean(scc_list),
            '4dmax': np.nanmean(_4dmax_scc_list),
            'optical_flow': np.nanmean(of_scc_list),
            'linear': np.nanmean(linear_scc_list)

        },
        "genome_disco": {
            'ours': np.nanmean(genome_disco_list),
            '4dmax': np.nanmean(_4dmax_genome_disco_list),
            'optical_flow': np.nanmean(of_genome_disco_list),
            'linear': np.nanmean(linear_genome_disco_list)

        },
        "genome_disco2": {
            'ours': np.nanmean(genome_disco2_list),
            '4dmax': np.nanmean(_4dmax_genome_disco2_list),
            'optical_flow': np.nanmean(of_genome_disco2_list),
            'linear': np.nanmean(linear_genome_disco2_list)
        },
        "hicrep": {
            'ours': np.nanmean(hicrep_list),
            '4dmax': np.nanmean(_4dmax_hicrep_list),
            'optical_flow': np.nanmean(of_hicrep_list),
            'linear': np.nanmean(linear_hicrep_list)
        }
    }

    return metrics


COLUMN_NAMES = ["model", "resolution", "patch", "organism", "sample", "condition", "time", "chromosome",
                "psnr_ours", "psnr_4dmax",  "psnr_optical_flow", "psnr_linear",
                "ssim_ours",  "ssim_4dmax", "ssim_optical_flow", "ssim_linear",
                "spearman_ours",  "spearman_4dmax", "spearman_optical_flow", "spearman_linear",
                "scc_ours",  "scc_4dmax", "scc_optical_flow", "scc_linear",
                "genome_disco_ours",  "genome_disco_4dmax", "genome_disco_optical_flow", "genome_disco_linear",
                "genome_disco2_ours",  "genome_disco2_4dmax", "genome_disco2_optical_flow", "genome_disco2_linear",
                "hicrep_ours",  "hicrep_4dmax", "hicrep_optical_flow", "hicrep_linear"
                ]


def write_summary(model, resolution, patch, organism, sample, condition, time, chromosome, metrics, output_file):
    row = [
        str(model),
        str(resolution),
        str(patch),
        organism,
        sample,
        condition,
        str(time),
        chromosome,
        f"{metrics['psnr']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['psnr']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['psnr']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['psnr']['linear']:.{METRIC_PRECISION}f}",

        f"{metrics['ssim']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['ssim']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['ssim']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['ssim']['linear']:.{METRIC_PRECISION}f}",

        f"{metrics['spearman']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['spearman']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['spearman']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['spearman']['linear']:.{METRIC_PRECISION}f}",

        f"{metrics['scc']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['scc']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['scc']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['scc']['linear']:.{METRIC_PRECISION}f}",

        f"{metrics['genome_disco']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco']['linear']:.{METRIC_PRECISION}f}",

        f"{metrics['genome_disco2']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco2']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco2']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco2']['linear']:.{METRIC_PRECISION}f}",

        f"{metrics['hicrep']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['hicrep']['4dmax']:.{METRIC_PRECISION}f}",
        f"{metrics['hicrep']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['hicrep']['linear']:.{METRIC_PRECISION}f}"
    ]

    with open(output_file, "a") as f:
        f.write(",".join(row) + "\n")
        f.flush()
        f.close()

    print(row)
    LOG.info(row)


def run_inference():
    for model_res in RESOLUTIONS:
        for model_patch, model_batch in zip(PATCHES, MODEL_BATCHES):
            config_name = f'config_{get_res_tag(model_res)}_{model_patch}'

            model_output_dir = os.path.join(OUTPUT_DIR, config_name)
            os.makedirs(model_output_dir, exist_ok=True)

            csv_filename = os.path.join(
                model_output_dir, f'{config_name}_{CSV_FILENAME_SUFFIX}')
            with open(csv_filename, "w") as f:
                f.write(",".join(COLUMN_NAMES) + "\n")

            heatmap_output_dir = os.path.join(
                model_output_dir, 'heatmaps')
            os.makedirs(heatmap_output_dir, exist_ok=True)

            model_filename = os.path.join(
                MODEL_DIR, config_name, f"hicinterpolate_{model_patch}_p{model_patch}_b{model_batch}.pt")
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
                                        DICT_DIR, f'test_{resolution}_{model_patch}_{organism}_{sample}_{subsample}_{triplet[1]}_chr{chromosome}.txt')
                                    print(
                                        f"Running inference for record: {record_filename}")
                                    LOG.info(
                                        f"Running inference for record: {record_filename}")

                                    plot_filename = f"{heatmap_output_dir}/plot_{resolution}_{model_patch}_{organism}_{sample}_{subsample}_{triplet[1]}_{chromosome}.png"
                                    metrics = get_prediction(batch_size=BATCH_SIZE, dataset_dict=record_filename, model=model,
                                                             heatmap_filename=plot_filename, resol=resolution, patch=model_patch)

                                    write_summary(model=config_name, resolution=resolution, patch=model_patch, organism=organism, sample=sample,
                                                  condition=subsample, time=triplet[1], chromosome=chromosome, metrics=metrics, output_file=csv_filename)


if __name__ == "__main__":
    run_inference()
