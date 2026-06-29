from time import time

from src.interpolator.model import Interpolator
from src.metric.hicrep import compute_hicrep
# from src.metric.genomedisco import compute_genomedisco
from src.metric import genome_disco as compute_genomedisco
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

ROOT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/HiCInterpolate'
MODEL_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/final_output/triplets_dataset'
DICT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset/test'
IMAGE_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset'

MODEL_SUBDIR = "config_25k_256"
MODEL_NAME = "hicinterpolate_256_p256_b20.pt"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [512, 256, 128, 64]
BATCHES = [10, 20, 20, 20]

OUTPUT_DIR = f'/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/final_output/HiCInterpolate/{MODEL_SUBDIR}'
CSV_FILENAME = os.path.join(
    OUTPUT_DIR, "log_comparison_hicinterpolate_diag.csv")

SUMMARY_FILENAME = os.path.join(OUTPUT_DIR, "log_comparison_summary_diag.csv")
OUTPUT_HEATMAP_DIR = os.path.join(OUTPUT_DIR, "pred_heatmaps")
CONFIG_PATH = f'/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/src/inference/config.yml'

METHODS = ("hicinterpolate", "linear", "optical_flow")
METRICS = ("psnr", "ssim", 'scc', "genome_disco", "hicrep", "lpips")

METRIC_PRECISION = 4
NUM_VIZ_SAMPLES = 2


CHROMOSOMES = {
    "human": ["10", "11", "15", "16", "20", "21"]
}

TEST_DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    ["4DNFIP9EJSOM_dmso_control_0m",
                     "4DNFI7T93SHL_dmso_control_30m",
                     "4DNFICF2Z2TG_dmso_control_60m"],

                    ["4DNFI7T93SHL_dmso_control_30m",
                     "4DNFICF2Z2TG_dmso_control_60m",
                     "4DNFILL624WG_dmso_control_90m"],

                    ["4DNFICF2Z2TG_dmso_control_60m",
                     "4DNFILL624WG_dmso_control_90m",
                     "4DNFIC4GB8UM_dmso_control_120m"]
                ]
            }
        },
        "dtag": {
            "v1": {
                "triplets":
                [
                    ["4DNFI5EAPQTI_dtag_v1_0m",
                     "4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m"],

                    ["4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m"],

                    ["4DNFIXWT5U42_dtag_v1_60m",
                     "4DNFIHTFIMGG_dtag_v1_90m",
                     "4DNFIPZCCTV6_dtag_v1_120m"]
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
    return default_collate(batch)


def get_dataloader(ds: Dataset, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=shuffle,
        worker_init_fn=set_seed,
        num_workers=20,
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
    is_drawn = False

    psnr_list = []
    ssim_list = []
    scc_list = []
    genome_disco_list = []
    hicrep_list = []
    lpips_list = []

    linear_psnr_list = []
    linear_ssim_list = []
    linear_scc_list = []
    linear_genome_disco_list = []
    linear_hicrep_list = []
    linear_lpips_list = []

    of_psnr_list = []
    of_ssim_list = []
    of_scc_list = []
    of_genome_disco_list = []
    of_hicrep_list = []
    of_lpips_list = []

    for _, (x0, y, x1) in enumerate(tqdm(test_dl)):
        x0 = x0.to(DEVICE)
        y = y.to(DEVICE)
        x1 = x1.to(DEVICE)
        pred = model(x0, x1)
        pred[pred < 0] = 0
        linear = linear_interpolation(x0, x1, t=0.5)
        of = OF(x0, x1)

        if is_drawn is False:
            num_examples = min(2, len(y))
            y_cpu = y[:num_examples]
            pred_cpu = pred[:num_examples]
            linear_cpu = linear[:num_examples]
            of_cpu = of[:num_examples]
            plot.draw_hic_comparison(num_examples=num_examples, target=y_cpu,
                                     pred=pred_cpu, linear=linear_cpu, of=of_cpu, file=heatmap_filename)
            is_drawn = True

        psnr_list.append(eval_metric.get_psnr_from_tensor(pred, y).item())
        ssim_list.append(eval_metric.get_ms_ssim_from_tensor(pred, y).item())
        scc_list.append(eval_metric.get_scc_from_tensor(pred, y).item())
        genome_disco_list.append(compute_genomedisco.compute_reproducibility_from_tensor(
            pred, y))
        hicrep_list.append(compute_hicrep.get_hicrep_scc_from_tensor(
            pred, y, resol=resol, h=5, lbr=0, ubr=resol*(patch-1)))
        lpips_list.append(eval_metric.get_lpips(pred, y).item())

        linear_psnr_list.append(
            eval_metric.get_psnr_from_tensor(linear, y).item())
        linear_ssim_list.append(
            eval_metric.get_ms_ssim_from_tensor(linear, y).item())
        linear_scc_list.append(
            eval_metric.get_scc_from_tensor(linear, y).item())
        linear_genome_disco_list.append(compute_genomedisco.compute_reproducibility_from_tensor(
            linear, y))
        linear_hicrep_list.append(compute_hicrep.get_hicrep_scc_from_tensor(
            linear, y, resol=resol, h=5, lbr=0, ubr=resol*(patch-1)))
        linear_lpips_list.append(
            eval_metric.get_lpips(linear, y).item())

        of_psnr_list.append(eval_metric.get_psnr_from_tensor(of, y).item())
        of_ssim_list.append(eval_metric.get_ms_ssim_from_tensor(of, y).item())
        of_scc_list.append(eval_metric.get_scc_from_tensor(of, y).item())
        of_genome_disco_list.append(compute_genomedisco.compute_reproducibility_from_tensor(
            of, y))
        of_hicrep_list.append(compute_hicrep.get_hicrep_scc_from_tensor(
            of, y, resol=resol, h=5, lbr=0, ubr=resol*(patch-1)))
        of_lpips_list.append(eval_metric.get_lpips(of, y).item())

        del x0, y, x1, pred, linear, of

    metrics = {
        "psnr": {'ours': np.nanmean(np.array(psnr_list)), 'linear': np.nanmean(np.array(linear_psnr_list)), 'optical_flow': np.nanmean(np.array(of_psnr_list))},
        "ssim": {'ours': np.nanmean(np.array(ssim_list)), 'linear': np.nanmean(np.array(linear_ssim_list)), 'optical_flow': np.nanmean(np.array(of_ssim_list))},
        "scc": {'ours': np.nanmean(np.array(scc_list)), 'linear': np.nanmean(np.array(linear_scc_list)), 'optical_flow': np.nanmean(np.array(of_scc_list))},
        "genome_disco": {'ours': np.nanmean(np.array(genome_disco_list)), 'linear': np.nanmean(np.array(linear_genome_disco_list)), 'optical_flow': np.nanmean(np.array(of_genome_disco_list))},
        "hicrep": {'ours': np.nanmean(np.array(hicrep_list)), 'linear': np.nanmean(np.array(linear_hicrep_list)), 'optical_flow': np.nanmean(np.array(of_hicrep_list))},
        "lpips": {'ours': np.nanmean(np.array(lpips_list)), 'linear': np.nanmean(np.array(linear_lpips_list)), 'optical_flow': np.nanmean(np.array(of_lpips_list))},
    }

    return metrics


COLUMN_NAMES = ["model", "resolution", "patch", "organism", "sample", "condition", "time", "chromosome",
                "psnr_ours", "psnr_linear", "psnr_optical_flow",
                "ssim_ours", "ssim_linear", "ssim_optical_flow",
                "scc_ours", "scc_linear", "scc_optical_flow",
                "genome_disco_ours", "genome_disco_linear", "genome_disco_optical_flow",
                "hicrep_ours", "hicrep_linear", "hicrep_optical_flow",
                "lpips_ours", "lpips_linear", "lpips_optical_flow"]


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
        f"{metrics['psnr']['linear']:.{METRIC_PRECISION}f}",
        f"{metrics['psnr']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['ssim']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['ssim']['linear']:.{METRIC_PRECISION}f}",
        f"{metrics['ssim']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['scc']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['scc']['linear']:.{METRIC_PRECISION}f}",
        f"{metrics['scc']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco']['linear']:.{METRIC_PRECISION}f}",
        f"{metrics['genome_disco']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['hicrep']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['hicrep']['linear']:.{METRIC_PRECISION}f}",
        f"{metrics['hicrep']['optical_flow']:.{METRIC_PRECISION}f}",
        f"{metrics['lpips']['ours']:.{METRIC_PRECISION}f}",
        f"{metrics['lpips']['linear']:.{METRIC_PRECISION}f}",
        f"{metrics['lpips']['optical_flow']:.{METRIC_PRECISION}f}"
    ]

    with open(output_file, "a") as f:
        f.write(",".join(row) + "\n")
        f.flush()
        f.close()


def run_inference():
    for model_res, model_patch, model_batch in zip(RESOLUTIONS, PATCHES, BATCHES):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_HEATMAP_DIR, exist_ok=True)

        with open(CSV_FILENAME, "w") as f:
            f.write(",".join(COLUMN_NAMES) + "\n")

        model_filename = os.path.join(
            MODEL_DIR, f'config_{get_res_tag(model_res)}_{model_patch}', f"hicinterpolate_{model_patch}_p{model_patch}_b{model_batch}.pt")
        model = load_model(model_filename)
        print(f"Running inference for model: {model_filename}")

        for resolution in RESOLUTIONS:
            for organism, samples in TEST_DATASET.items():
                for sample, condition in samples.items():
                    for subsample, content in condition.items():
                        for triplet in content["triplets"]:
                            uuid = triplet[0] + "_" + \
                                triplet[1] + "_" + triplet[2]
                            for chromosome in CHROMOSOMES[organism]:
                                record_filename = os.path.join(
                                    DICT_DIR, f"test_{str(resolution)}_{str(model_patch)}_{organism}_{triplet[1]}_{chromosome}.txt")
                                print(
                                    f"Running inference for record: {record_filename}")
                                metrics = get_prediction(batch_size=model_batch, dataset_dict=record_filename, model=model,
                                                         heatmap_filename=os.path.join(OUTPUT_HEATMAP_DIR, f"pred_heatmap_{str(resolution)}_{str(model_patch)}_{organism}_{triplet[1]}_{chromosome}.png"), resol=resolution, patch=model_patch)

                                write_summary(model=f'config_{get_res_tag(model_res)}_{model_patch}', resolution=resolution, patch=model_patch, organism=organism, sample=sample,
                                              condition=subsample, time=triplet[1], chromosome=chromosome, metrics=metrics, output_file=CSV_FILENAME)
                                print(
                                    f"PSNR={metrics['psnr']['ours']:.4f}, SSIM={metrics['ssim']['ours']:.4f}, SCC={metrics['scc']['ours']:.4f}, GenomeDisco={metrics['genome_disco']['ours']:.4f}, HiCRep={metrics['hicrep']['ours']:.4f}, LPIPS={metrics['lpips']['ours']:.4f}")


if __name__ == "__main__":
    run_inference()
