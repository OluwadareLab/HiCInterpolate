import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.metric.metrics import (
    get_genome_disco_gpu,
    get_hicrep_gpu,
    get_lpips_gpu,
    get_psnr_gpu,
    get_ssim_gpu,
)
from src.inference import InfConfig, InfCustomDataset
from src import InferenceLib
from src.feature_encoder import FeatureEncoder
from flow_based_interpolation import of_interpolation
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import torch.distributed as dist
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import csv
import glob
import logging
import os
import random
import re
import sys
from statistics import mean
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate"
MODEL_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/output/log_mm_triplets_dataset"
DICT_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/log_mm_triplets_dataset/test"
IMAGE_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/log_mm_triplets_dataset"
OUTPUT_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/test_result"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "comparison_hicinterpolate_diag.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "comparison_summary_diag.csv")
OUTPUT_HEATMAP_DIR = os.path.join(OUTPUT_DIR, "pred_heatmaps")
MANIFEST_FILE = os.path.join(OUTPUT_HEATMAP_DIR, "manifest.csv")
CONFIG_PATH = f"{ROOT_DIR}/HiCInterpolate/src/inference/config.yml"

RESULT_ID_COLS = [
    "resolution",
    "patch_size",
    "organism",
    "sample",
    "subsample",
    "frame_uuid",
    "chromosome",
]
METHODS = ("hicinterpolate", "linear", "optical_flow")
METRICS = ("psnr", "ssim", "genome_disco", "hicrep", "lpips")
METRIC_PRECISION = 4
NUM_VIZ_SAMPLES = 2


def build_metric_fields() -> List[str]:
    return [
        f"{metric}_{method}"
        for metric in METRICS
        for method in METHODS
    ]

RESOLUTIONS = [25000]
BATCHES = [20]
PATCHES = [128]

CHROMOSOMES = {
    "human": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22"],
    "mouse": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
              "17", "18", "19"]
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
                    ["4DNFI5EAPQTI_dtag_v1_0m",
                     "4DNFIY1TCVLX_dtag_v1_30m",
                     "4DNFIXWT5U42_dtag_v1_60m"]
                ]
            }
        },

        "hct116": {
            "1": {
                "triplets":
                [
                    ["4DNFIDBFENL7_hct116_1_20m",
                     "4DNFI9ZUXG61_hct116_1_40m",
                     "4DNFIAUMRM2S_hct116_1_60m"]
                ]
            },

            "2": {
                "triplets":
                [
                    ["4DNFITUPI4HA_hct116_2_no_atp_120m_20m",
                     "4DNFIM7Q2FQQ_hct116_2_no_atp_120m_40m",
                     "4DNFISATK9PF_hct116_2_no_atp_120m_60m"]
                ]
            },
        },

        "hela_s3": {
            "r1": {
                "triplets":
                [
                    ["4DNFIEQHTV1R_hela_s3_r1_210m",
                     "4DNFIFW7GA64_hela_s3_r1_240m",
                     "4DNFIXGXD67I_hela_s3_r1_270m"]
                ]
            },

            "r2": {
                "triplets":
                [
                    ["4DNFIMD9QNDX_hela_s3_r2_210m",
                     "4DNFIATA1HD5_hela_s3_r2_240m",
                     "4DNFIH9U4I7I_hela_s3_r2_270m"]
                ]
            },

            "r3": {
                "triplets":
                [
                    ["4DNFI2KM22QR_hela_s3_r3_210m",
                     "4DNFIVF8Q45U_hela_s3_r3_240m",
                     "4DNFI2RN3WFP_hela_s3_r3_270m"]
                ]
            }
        }
    },

    "mouse": {
        "embryo": {
            "development": {
                "triplets": [
                    ["4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell"],

                    ["4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm",
                     "4DNFI5IAH9H1_mes_cell"]
                ]
            }
        }
    }
}

CMAP_JUICEBOX = mcolors.LinearSegmentedColormap.from_list(
    "juicebox", ["#fee8c8", "#fdbb84", "#e34a33", "#b30000"], N=256
)
MANIFEST_FIELDS = RESULT_ID_COLS + ["patch_index", "output_path"]
_MODEL_CACHE: Dict[str, tuple] = {}
_BASE_CFG = None
_LOG = None


class LegacyDecoderBlock(nn.Module):
    def __init__(self, in_channels=16, skip_channels=32, out_channels=32):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.comb = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, deep_ftr, skip_connection):
        x = self.upsample(deep_ftr)
        x = torch.cat([x, skip_connection], dim=1)
        return self.comb(x)


class LegacyFeatureDecoder(nn.Module):
    def __init__(self, cfg, feature_channels=None, out_channels=1):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = list(feature_channels or [32, 64, 128, 256, 512])
        self.level1 = LegacyDecoderBlock(self.feature_channels[1], self.feature_channels[0], self.feature_channels[0])
        self.level2 = LegacyDecoderBlock(self.feature_channels[2], self.feature_channels[1], self.feature_channels[1])
        self.level3 = LegacyDecoderBlock(self.feature_channels[3], self.feature_channels[2], self.feature_channels[2])
        self.level4 = LegacyDecoderBlock(self.feature_channels[4], self.feature_channels[3], self.feature_channels[3])

    def forward(self, ftr_stk: List[torch.Tensor]) -> torch.Tensor:
        out = self.level4(ftr_stk[4], ftr_stk[3])
        out = self.level3(out, ftr_stk[2])
        out = self.level2(out, ftr_stk[1])
        return self.level1(out, ftr_stk[0])


class LegacyInterpolator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            from src.flow_predictor import BackwardFlow, ForwardFlow
        except ImportError as exc:
            raise ImportError(
                "LegacyInterpolator requires BackwardFlow/ForwardFlow"
            ) from exc
        self.cfg = cfg
        self.in_channels = 1
        self.feature_channels = [32, 64, 128, 256, 512]
        self.out_channels = 1
        self.feature_encoder = FeatureEncoder(cfg, in_channels=self.in_channels, out_channels=self.feature_channels)
        self.forward_flow = ForwardFlow(cfg, feature_channels=self.feature_channels)
        self.backward_flow = BackwardFlow(cfg, feature_channels=self.feature_channels)
        self.feature_decoder = LegacyFeatureDecoder(cfg, feature_channels=self.feature_channels)
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feature_channels[0], kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(self.feature_channels[0]),
            nn.Conv2d(self.feature_channels[0], self.feature_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feature_channels[0]),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.feature_channels[0], self.feature_channels[0] // 2, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.BatchNorm2d(self.feature_channels[0] // 2),
            nn.Conv2d(self.feature_channels[0] // 2, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    @staticmethod
    def concatenate_flow_ftr(ftr_0: List[torch.Tensor], ftr_2: List[torch.Tensor]) -> List[torch.Tensor]:
        return [0.5 * (feature1 + feature2) for feature1, feature2 in zip(ftr_0, ftr_2)]

    def forward(self, x0: torch.Tensor, x2: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x0 = self.in_proj(x0)
        x2 = self.in_proj(x2)
        ftrs0 = self.feature_encoder(x0)
        ftrs2 = self.feature_encoder(x2)
        forward_mid_ftrs = self.forward_flow(ftrs0, ftrs2, time[:, 0])
        backward_mid_ftrs = self.backward_flow(ftrs2, ftrs0, time[:, 0])
        mid_ftrs = self.concatenate_flow_ftr(forward_mid_ftrs, backward_mid_ftrs)
        residual = self.feature_decoder(mid_ftrs)
        return self.out_proj(residual + mid_ftrs[0])


def load_runner_model(cfg, log, model_path: str, dl: DataLoader, isDistributed: bool = False):
    try:
        runner = InferenceLib.HiCInterpolate(
            cfg=cfg, log=log, model=model_path, dl=dl, isDistributed=isDistributed
        )
        return runner._get_model()
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        print("[WARN] Current Interpolator shape mismatches checkpoint; retrying legacy decoder path.")
        original_interpolator = InferenceLib.Interpolator
        try:
            InferenceLib.Interpolator = LegacyInterpolator
            runner = InferenceLib.HiCInterpolate(
                cfg=cfg, log=log, model=model_path, dl=dl, isDistributed=isDistributed
            )
            return runner._get_model()
        finally:
            InferenceLib.Interpolator = original_interpolator



def base_logger(file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    logger = logging.getLogger(os.path.basename(file))
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(file)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


def set_seed(seed_v: int = 42):
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def get_dataloader(ds: Dataset, batch_size: int = 8, isDistributed: bool = False) -> DataLoader:
    kwargs = dict(batch_size=batch_size,
                  collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    if isDistributed:
        return DataLoader(
            ds,
            **kwargs,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(ds, shuffle=False),
        )
    return DataLoader(ds, **kwargs, shuffle=False, worker_init_fn=set_seed)


def shutdown_dataloader(dl: DataLoader) -> None:
    iterator = getattr(dl, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def resolution_tag(resolution: int) -> Optional[str]:
    return {25000: "25k", 10000: "10k", 5000: "5k"}.get(resolution)


def build_job_cfg(config_filename: str, dataset_filename: str, model_path: str, batch_size: int):
    global _BASE_CFG
    if _BASE_CFG is None:
        yaml_cfg = OmegaConf.load(CONFIG_PATH)
        _BASE_CFG = OmegaConf.merge(OmegaConf.structured(InfConfig), yaml_cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(_BASE_CFG, resolve=True))
    OmegaConf.update(cfg, "file.inference", dataset_filename)
    OmegaConf.update(cfg, "file.model", model_path)
    OmegaConf.update(cfg, "dir.image", IMAGE_DIR)
    OmegaConf.update(cfg, "data.patch", PATCHES[0])
    OmegaConf.update(cfg, "data.batch_size", batch_size)
    OmegaConf.update(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_state_dir, exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)
    OmegaConf.update(cfg, "dir.model_state", model_state_dir)
    return cfg


def get_or_load_model(cfg, log, model_path: str, dl: DataLoader, isDistributed: bool = False):
    cache_key = f"{model_path}|{cfg.device}|{isDistributed}"
    if cache_key not in _MODEL_CACHE:
        model, device = load_runner_model(cfg, log, model_path, dl, isDistributed=isDistributed)
        model.eval()
        _MODEL_CACHE[cache_key] = (model, device)
    return _MODEL_CACHE[cache_key]


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return (1.0 - t) * x0 + t * x1


def random_patch_indices(n: int, k: int = NUM_VIZ_SAMPLES) -> List[int]:
    if n <= 0:
        return []
    return sorted(random.sample(range(n), min(k, n)))


def prepare_matrix(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).squeeze().copy()
    mat[mat <= 0] = np.nan
    return mat


def log_norm_range(*matrices: np.ndarray) -> LogNorm:
    finite_parts = [m[np.isfinite(m)] for m in matrices if m.size > 0]
    if not finite_parts:
        return LogNorm(vmin=1e-3, vmax=1.0)
    combined = np.concatenate(finite_parts)
    if combined.size == 0:
        return LogNorm(vmin=1e-3, vmax=1.0)
    vmin = np.nanpercentile(combined, 5)
    vmax = np.nanpercentile(combined, 99)
    if vmin <= 0 or not np.isfinite(vmin):
        vmin = np.nanmin(combined)
    if vmax <= vmin or not np.isfinite(vmax):
        vmax = vmin + 1.0
    return LogNorm(vmin=max(vmin, 1e-8), vmax=max(vmax, vmin + 1e-8))


def heatmap_path(job_meta: dict, patch_index: int) -> str:
    res_dir = f"res_{job_meta['resolution']}_patch_{job_meta['patch_size']}"
    chr_dir = f"chr{job_meta['chromosome']}"
    patch_name = f"patch_{patch_index:05d}_6panels.png"
    rel = os.path.join(
        res_dir,
        job_meta["organism"],
        job_meta["sample"],
        job_meta["subsample"],
        job_meta["frame"],
        chr_dir,
        patch_name,
    )
    return os.path.join(OUTPUT_HEATMAP_DIR, rel)


def save_comparison_heatmap(
    x0, y, pred_hic, pred_linear, pred_of, x1, out_path: str
) -> None:
    mats = [
        prepare_matrix(m)
        for m in (x0, y, pred_hic, pred_linear, pred_of, x1)
    ]
    norm = log_norm_range(*mats)
    fig, axes = plt.subplots(1, 6, figsize=(24, 4), dpi=300)
    panels = zip(
        axes,
        mats,
        ("x_0", "Ground truth", "HiCInterpolate", "Linear", "Optical flow", "x_1"),
    )
    for ax, matrix, title in panels:
        im = ax.imshow(matrix, cmap=CMAP_JUICEBOX,
                       norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6, length=2)
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def append_csv(row: dict, path: str, fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        fid.flush()
        os.fsync(fid.fileno())


def update_summary() -> None:
    if not os.path.exists(OUTPUT_FILE):
        return
    df = pd.read_csv(OUTPUT_FILE)
    metric_cols = [col for col in build_metric_fields() if col in df.columns]
    if not metric_cols:
        return
    summary = df.groupby(
        ["resolution", "patch_size", "organism"], dropna=False
    )[metric_cols].mean()
    summary.round(METRIC_PRECISION).reset_index().to_csv(
        SUMMARY_FILE, index=False)


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() > 1:
            value = value.mean()
        value = value.item()
    return float(value)


def eval_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "psnr": scalar(get_psnr_gpu(pred, target)),
        "ssim": scalar(get_ssim_gpu(pred, target)),
        "genome_disco": scalar(get_genome_disco_gpu(pred, target)),
        "hicrep": scalar(get_hicrep_gpu(pred, target)),
        "lpips": scalar(get_lpips_gpu(pred, target)),
    }


@torch.no_grad()
def process_chromosome(cfg, model, device, ds, batch_size: int, job_meta: dict, save_plots: bool) -> dict:
    dl = get_dataloader(ds=ds, batch_size=batch_size, isDistributed=False)
    metrics = {method: {metric: [] for metric in METRICS}
               for method in METHODS}
    plot_indices = set(random_patch_indices(
        len(ds), NUM_VIZ_SAMPLES)) if save_plots else set()
    manifest_fields = MANIFEST_FIELDS
    saved_plots = 0
    seen = 0

    try:
        model.eval()
        desc = f"chr{job_meta['chromosome']}"
        for batch in tqdm(dl, desc=desc, leave=False):
            if batch is None:
                continue
            x0, y, x1, time_frame = batch
            x0 = x0.to(device)
            x1 = x1.to(device)
            y = y.to(device)
            time_frame = time_frame.to(device)

            pred_hic = model(x0, x1, time_frame)
            t_val = float(time_frame.view(-1)[0].item())
            pred_linear = linear_interpolation(x0, x1, t=t_val)
            pred_of = of_interpolation(x0, x1)

            preds = {
                "hicinterpolate": pred_hic,
                "linear": pred_linear,
                "optical_flow": pred_of,
            }
            for method, pred in preds.items():
                batch_metrics = eval_all_metrics(pred, y)
                for metric_name, metric_value in batch_metrics.items():
                    if np.isfinite(metric_value):
                        metrics[method][metric_name].append(metric_value)

            for local_idx in range(y.size(0)):
                patch_index = seen + local_idx
                if patch_index not in plot_indices:
                    continue
                out_path = heatmap_path(job_meta, patch_index)
                save_comparison_heatmap(
                    x0[local_idx].detach().cpu().numpy(),
                    y[local_idx].detach().cpu().numpy(),
                    pred_hic[local_idx].detach().cpu().numpy(),
                    pred_linear[local_idx].detach().cpu().numpy(),
                    pred_of[local_idx].detach().cpu().numpy(),
                    x1[local_idx].detach().cpu().numpy(),
                    out_path,
                )
                append_csv(
                    {**job_meta, "patch_index": patch_index,
                        "output_path": out_path},
                    MANIFEST_FILE,
                    manifest_fields,
                )
                saved_plots += 1
            seen += y.size(0)

            del x0, y, x1, time_frame, pred_hic, pred_linear, pred_of
    finally:
        shutdown_dataloader(dl)

    row = {key: job_meta[key] for key in RESULT_ID_COLS}
    for metric_name in METRICS:
        for method in METHODS:
            values = metrics[method][metric_name]
            col = f"{metric_name}_{method}"
            row[col] = round(
                mean(values), METRIC_PRECISION) if values else np.nan
    row["_num_heatmaps"] = saved_plots
    return row


def iter_known_jobs(organism_filter: Optional[str], chromosome_filter: Optional[str]):
    for resolution in RESOLUTIONS:
        res_tag = resolution_tag(resolution)
        if res_tag is None:
            print(f"[WARN] Unsupported resolution: {resolution}")
            continue
        for patch in PATCHES:
            for batch_size in BATCHES:
                for organism, samples in TEST_DATASET.items():
                    if organism_filter and organism != organism_filter:
                        continue
                    for sample, subsamples in samples.items():
                        for subsample, content in subsamples.items():
                            for triplet in content["triplets"]:
                                frame_uuid = triplet[1]
                                chromosomes = CHROMOSOMES[organism]
                                if chromosome_filter:
                                    chromosomes = [
                                        c for c in chromosomes if c == chromosome_filter]
                                for chromosome in chromosomes:
                                    yield {
                                        "resolution": resolution,
                                        "patch_size": patch,
                                        "batch_size": batch_size,
                                        "organism": organism,
                                        "sample": sample,
                                        "subsample": subsample,
                                        "frame": frame_uuid,
                                        "chromosome": chromosome,
                                        "res_tag": res_tag,
                                    }



def split_triplet_uuid(frame_uuid: str) -> List[str]:
    starts = [0] + [match.start() + 1 for match in re.finditer(r"_4DNFI", frame_uuid)]
    if len(starts) != 3:
        return [frame_uuid]
    starts.append(len(frame_uuid) + 1)
    return [frame_uuid[starts[i]:starts[i + 1] - 1] for i in range(3)]


def infer_sample_meta(organism: str, triplet: List[str]) -> tuple[str, str]:
    middle = triplet[1] if len(triplet) > 1 else triplet[0]
    parts = middle.split("_")
    if len(parts) < 3:
        return organism, "unknown"
    if "dmso" in parts:
        return "dmso", "control"
    if "hct116" in parts:
        idx = parts.index("hct116")
        return "hct116", parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    if "hela" in parts and "s3" in parts:
        idx = parts.index("hela")
        replicate = parts[idx + 2] if idx + 2 < len(parts) else "unknown"
        return "hela_s3", replicate
    sample = "_".join(parts[1:-1]) or organism
    subsample = parts[-2] if len(parts) >= 4 else "unknown"
    return sample, subsample


def resolve_model_path(res_tag: str, patch: int, batch_size: int) -> Optional[str]:
    model_name = f"hicinterpolate_{patch}_p{patch}_b{batch_size}.pt"
    subdir_candidates = [
        f"config_dilated_{res_tag}_{patch}",
        f"config_dilated_{res_tag}_p{patch}",
        f"config_a1_{res_tag}_p{patch}_b{batch_size}",
    ]
    for subdir in subdir_candidates:
        model_path = os.path.join(MODEL_DIR, subdir, model_name)
        if os.path.exists(model_path):
            return model_path
    return None


def record_path(job: dict) -> str:
    return os.path.join(
        DICT_DIR,
        (
            "test_{}_{}_{}_{}_{}.txt".format(
                job["resolution"],
                job["patch_size"],
                job["organism"],
                job["frame"],
                job["chromosome"],
            )
        ),
    )


def iter_record_jobs(organism_filter: Optional[str], chromosome_filter: Optional[str]):
    seen = set()
    for job in iter_known_jobs(organism_filter, chromosome_filter):
        path = record_path(job)
        if not os.path.exists(path):
            continue
        key = (job["resolution"], job["patch_size"], job["organism"], job["frame"], job["chromosome"])
        seen.add(key)
        yield {**job, "record_file": path}

    pattern = os.path.join(DICT_DIR, "test_*_*.txt")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        match = re.match(r"^test_(\d+)_(\d+)_(human|mouse)_(.+)_([^_]+)\.txt$", name)
        if not match:
            continue
        resolution = int(match.group(1))
        patch = int(match.group(2))
        organism = match.group(3)
        frame_uuid = match.group(4)
        chromosome = match.group(5)
        if resolution not in RESOLUTIONS or patch not in PATCHES:
            continue
        if organism_filter and organism != organism_filter:
            continue
        if chromosome_filter and chromosome != chromosome_filter:
            continue
        res_tag = resolution_tag(resolution)
        if res_tag is None:
            continue
        key = (resolution, patch, organism, frame_uuid, chromosome)
        if key in seen:
            continue
        triplet = split_triplet_uuid(frame_uuid)
        sample, subsample = infer_sample_meta(organism, triplet)
        yield {
            "resolution": resolution,
            "patch_size": patch,
            "batch_size": BATCHES[0],
            "organism": organism,
            "sample": sample,
            "subsample": subsample,
            "frame": frame_uuid,
            "chromosome": chromosome,
            "res_tag": res_tag,
            "record_file": path,
        }


def run_inference(
    organism_filter: Optional[str] = None,
    chromosome_filter: Optional[str] = None,
    save_plots: bool = True,
    overwrite: bool = False,
) -> None:
    global _LOG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_HEATMAP_DIR, exist_ok=True)
    if overwrite:
        for path in (OUTPUT_FILE, SUMMARY_FILE, MANIFEST_FILE):
            if os.path.exists(path):
                os.remove(path)
    if _LOG is None:
        _LOG = base_logger(os.path.join(
            OUTPUT_DIR, "test_hicinterpolate_diag.log"))

    metric_fields = RESULT_ID_COLS + build_metric_fields()
    processed = 0
    skipped = 0
    discovered = 0

    for job in iter_record_jobs(organism_filter, chromosome_filter):
        ds_dict_filename = job["record_file"]
        discovered += 1
        model_path = resolve_model_path(
            job["res_tag"], job["patch_size"], job["batch_size"]
        )
        print(f"Processing {ds_dict_filename}")
        if not os.path.exists(ds_dict_filename):
            print(f"[WARN] Missing input file: {ds_dict_filename}")
            skipped += 1
            continue
        if model_path is None:
            print(
                f"[WARN] Missing model for res={job['res_tag']} "
                f"patch={job['patch_size']} batch={job['batch_size']}"
            )
            skipped += 1
            continue

        cfg = build_job_cfg("config", ds_dict_filename,
                            model_path, job["batch_size"])
        cds = InfCustomDataset(
            record_file=cfg.file.inference,
            img_dir=cfg.dir.image,
            img_map=cfg.data.interpolator_images_map,
        )
        ds = cds._get_inference_dl()
        if len(ds) == 0:
            print(f"[WARN] Empty dataset: {ds_dict_filename}")
            skipped += 1
            del ds, cds
            continue

        dl_for_model = get_dataloader(
            ds=ds, batch_size=job["batch_size"], isDistributed=False)
        try:
            model, device = get_or_load_model(
                cfg, _LOG, model_path, dl_for_model)
            job_meta = {key: job[key] for key in RESULT_ID_COLS if key != "frame_uuid"}
            job_meta["frame_uuid"] = job["frame"]
            job_meta["frame"] = job["frame"]
            row = process_chromosome(
                cfg, model, device, ds, job["batch_size"], job_meta, save_plots=save_plots
            )
            append_csv(row, OUTPUT_FILE, metric_fields)
            update_summary()
            processed += 1
            print(
                f"[OK] chr{job['chromosome']} metrics -> {OUTPUT_FILE}; "
                f"plots={row['_num_heatmaps']}"
            )
        finally:
            shutdown_dataloader(dl_for_model)
            del ds, cds, dl_for_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if discovered == 0:
        print(f"[WARN] No matching records in {DICT_DIR} for organism={organism_filter} chromosome={chromosome_filter}")
    print(
        f"Finished. processed={processed}, skipped={skipped}, "
        f"metrics={OUTPUT_FILE}, summary={SUMMARY_FILE}, manifest={MANIFEST_FILE}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HiCInterpolate diag metrics and save heatmaps chromosome-by-chromosome."
    )
    parser.add_argument("--organism", type=str, default=None,
                        help="Limit to one organism.")
    parser.add_argument("--chromosome", type=str,
                        default=None, help="Limit to one chromosome.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip heatmap generation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove old CSV outputs first.")
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    try:
        run_inference(
            organism_filter=args.organism,
            chromosome_filter=args.chromosome,
            save_plots=not args.no_plots,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
        raise
