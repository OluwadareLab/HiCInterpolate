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
import pandas as pd
import numpy as np
import argparse
import csv
import logging
import os
import random
import sys
from statistics import mean
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate"
MODEL_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/output/log_mm_triplets_dataset"
DICT_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/mm_triplets_dataset/test"
IMAGE_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/mm_triplets_dataset"
OUTPUT_DIR = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/output/test/mm_test"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "comparison_hicinterpolate_diag.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "comparison_summary_diag.csv")
OUTPUT_HEATMAP_DIR = os.path.join(OUTPUT_DIR, "pred_heatmaps")
MANIFEST_FILE = os.path.join(OUTPUT_HEATMAP_DIR, "manifest.csv")
CONFIG_PATH = f"{ROOT_DIR}/HiCInterpolate/src/inference/config.yml"

RESULT_ID_COLS = [
    "resolution",
    "patch_size",
    "batch_size",
    "organism",
    "sample",
    "subsample",
    "frame",
    "chromosome",
]
METHODS = ("hicinterpolate", "linear", "optical_flow")
METRICS = ("psnr", "ssim", "genome_disco", "hicrep", "lpips")
METRIC_PRECISION = 4
NUM_VIZ_SAMPLES = 2

RESOLUTIONS = [25000]
BATCHES = [30]
PATCHES = [128]

CHROMOSOMES = {
    "human": [
        "11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22"],
    "mouse": [
        "11", "12", "13", "14", "15", "16",
              "17", "18", "19"],
}

TEST_DATASET = {
    "human": {
        "dmso": {"control": {"triplets": [[
            "4DNFI7T93SHL_dmso_control_30m",
            "4DNFICF2Z2TG_dmso_control_60m",
            "4DNFILL624WG_dmso_control_90m",
        ]]}},
        "dtag": {"v1": {"triplets": [[
            "4DNFIY1TCVLX_dtag_v1_30m",
            "4DNFIXWT5U42_dtag_v1_60m",
            "4DNFIHTFIMGG_dtag_v1_90m",
        ]]}},
        "hct116": {
            "1": {"triplets": [[
                "4DNFIDBFENL7_hct116_1_20m",
                "4DNFI9ZUXG61_hct116_1_40m",
                "4DNFIAUMRM2S_hct116_1_60m",
            ]]},
            "2": {"triplets": [[
                "4DNFIAAH19VM_hct116_2_20m",
                "4DNFI7QUSU5J_hct116_2_40m",
                "4DNFIXEB4UZO_hct116_2_60m",
            ]]},
        },
        "hela_s3": {
            "r2": {"triplets": [[
                "4DNFIX6ZXCA8_hela_s3_r2_30m",
                "4DNFIEVR81FS_hela_s3_r2_60m",
                "4DNFIAUI6BBI_hela_s3_r2_90m",
            ]]},
            "r3": {"triplets": [[
                "4DNFICFZGFAV_hela_s3_r3_30m",
                "4DNFIQXCZVVA_hela_s3_r3_60m",
                "4DNFIB6PJFJ3_hela_s3_r3_90m",
            ]]},
        },
    }
}

CMAP_JUICEBOX = mcolors.LinearSegmentedColormap.from_list(
    "juicebox", ["#fee8c8", "#fdbb84", "#e34a33", "#b30000"], N=256
)
MANIFEST_FIELDS = RESULT_ID_COLS + ["patch_index", "output_path"]
_MODEL_CACHE: Dict[str, tuple] = {}
_BASE_CFG = None
_LOG = None


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
                  collate_fn=collate_fn, pin_memory=True)
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
    OmegaConf.update(cfg, "data.batch_size", batch_size)
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
        runner = InferenceLib.HiCInterpolate(
            cfg=cfg, log=log, model=model_path, dl=dl, isDistributed=isDistributed
        )
        model, device = runner._get_model()
        model.eval()
        _MODEL_CACHE[cache_key] = (model, device)
    return _MODEL_CACHE[cache_key]


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return (1.0 - t) * x0 + t * x1


def middle_indices(n: int, k: int = NUM_VIZ_SAMPLES) -> List[int]:
    if n <= 0:
        return []
    if n == 1 or k == 1:
        return [n // 2]
    mid = n // 2
    return sorted({max(0, mid - 1), min(n - 1, mid)})[:k]


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
    patch_name = f"patch_{patch_index:05d}_4methods.png"
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


def save_methods_heatmaps(y, pred_hic, pred_linear, pred_of, out_path: str) -> None:
    mats = [prepare_matrix(m) for m in (y, pred_hic, pred_linear, pred_of)]
    norm = log_norm_range(*mats)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    panels = zip(
        axes,
        mats,
        ("Ground truth", "HiCInterpolate", "Linear", "Optical flow"),
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
    metric_cols = [col for col in df.columns if any(
        col.endswith(f"_{m}") for m in METRICS)]
    if not metric_cols:
        return
    summary = df.groupby(["resolution", "patch_size", "batch_size",
                         "organism"], dropna=False)[metric_cols].mean()
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
    plot_indices = set(middle_indices(
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
                save_methods_heatmaps(
                    y[local_idx].detach().cpu().numpy(),
                    pred_hic[local_idx].detach().cpu().numpy(),
                    pred_linear[local_idx].detach().cpu().numpy(),
                    pred_of[local_idx].detach().cpu().numpy(),
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

    row = dict(job_meta)
    row["num_patches"] = seen
    row["num_heatmaps"] = saved_plots
    for method in METHODS:
        for metric_name in METRICS:
            values = metrics[method][metric_name]
            col = f"{method}_{metric_name}"
            row[col] = round(
                mean(values), METRIC_PRECISION) if values else np.nan
    return row


def iter_jobs(organism_filter: Optional[str], chromosome_filter: Optional[str]):
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
                                frame_uuid = '_'.join(triplet)
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

    metric_fields = RESULT_ID_COLS + ["num_patches", "num_heatmaps"] + [
        f"{method}_{metric}" for method in METHODS for metric in METRICS
    ]
    processed = 0
    skipped = 0

    for job in iter_jobs(organism_filter, chromosome_filter):
        ds_dict_filename = os.path.join(
            DICT_DIR,
            (
                f"test_{job['resolution']}_{job['patch_size']}_"
                f"{job['organism']}_{job['frame']}_{job['chromosome']}.txt"
            ),
        )
        # model_subdir = f"config_a1_{job['res_tag']}_p{job['patch_size']}_b{job['batch_size']}"
        # model_name = (
        #     f"hicinterpolate_{job['patch_size']}_"
        #     f"p{job['patch_size']}_b{job['batch_size']}.pt"
        # )
        model_subdir = "config_dilated_25k_128"
        model_name = "hicinterpolate_128_p128_b30.pt"
        model_path = os.path.join(MODEL_DIR, model_subdir, model_name)
        print(f"Processing {ds_dict_filename}")
        if not os.path.exists(ds_dict_filename):
            print(f"[WARN] Missing input file: {ds_dict_filename}")
            skipped += 1
            continue
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model: {model_path}")
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
            job_meta = {key: job[key] for key in RESULT_ID_COLS}
            row = process_chromosome(
                cfg, model, device, ds, job["batch_size"], job_meta, save_plots=save_plots
            )
            append_csv(row, OUTPUT_FILE, metric_fields)
            update_summary()
            processed += 1
            print(
                f"[OK] chr{job['chromosome']} metrics -> {OUTPUT_FILE}; "
                f"plots={row['num_heatmaps']}"
            )
        finally:
            shutdown_dataloader(dl_for_model)
            del ds, cds, dl_for_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
