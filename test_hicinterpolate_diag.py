from tqdm.asyncio import tqdm

import os
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import random
import sys
import logging
from torch.utils.data.dataloader import default_collate
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.inference import InfConfig, InfCustomDataset
from src import InferenceLib
from flow_based_interpolation import of_interpolation

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate"
MODEL_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/output/mm_triplets_dataset"
DICT_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/test_triplets/mm_triplets/diag_test"
IMAGE_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/mm_triplets_dataset"
OUTPUT_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/output/test"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "comparison_hicinterpolate_diag.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "comparison_summary_diag.csv")
CONFIG_PATH = f"{ROOT_DIR}/HiCInterpolate/src/inference/config.yml"
RESULT_ID_COLS = ["resolution", "patch_size", "organism",
                  "sample", "subsample", "frame", "chromosome"]
METRIC_PRECISION = 4

RESOLUTIONS = [25000, 10000]
BATCHES = [64]
PATCHES = [128, 64]

CHROMOSOMES = {
    "human": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22", "X", "Y"],
    "mouse": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "12", "13", "14", "15", "16",
              "17", "18", "19", "X", "Y"]
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
                    ["4DNFIAAH19VM_hct116_2_20m",
                     "4DNFI7QUSU5J_hct116_2_40m",
                     "4DNFIXEB4UZO_hct116_2_60m"]
                ]
            },
        },
        "hela_s3": {
            "r2": {
                "triplets":
                [
                    ["4DNFIX6ZXCA8_hela_s3_r2_30m",
                     "4DNFIEVR81FS_hela_s3_r2_60m",
                     "4DNFIAUI6BBI_hela_s3_r2_90m"]
                ]
            },
            "r3": {
                "triplets":
                [
                    ["4DNFICFZGFAV_hela_s3_r3_30m",
                     "4DNFIQXCZVVA_hela_s3_r3_60m",
                     "4DNFIB6PJFJ3_hela_s3_r3_90m"]
                ]
            }
        }
    },
    "mouse": {
        "embryo": {
            "development": {
                "triplets": [
                    ["4DNFIN8F14CS_sperm",
                     "4DNFIVCJKHMN_mii_oocyte",
                     "4DNFI1EYIGOC_zygote"],

                    ["4DNFIVCJKHMN_mii_oocyte",
                     "4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell"],

                    ["4DNFI1EYIGOC_zygote",
                     "4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell"],

                    ["4DNFIK4CECUH_early2_cell",
                     "4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell"],

                    ["4DNFICXCFGEI_late2_cell",
                     "4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm"],

                    ["4DNFIFA89L5B_8cell",
                     "4DNFIK5HY1GP_icm",
                     "4DNFI5IAH9H1_mes_cell"]
                ]
            }
        }
    }
}

_MODEL_CACHE = {}
_BASE_CFG = None
_LOG = None


def base_logger(file):
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            filename=file,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
    return logger


def set_seed(seed_v: int = 42):
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


def _num_dataloader_workers() -> int:
    # Each job builds a new DataLoader; avoid many long-lived worker pools.
    return min(4, os.cpu_count() or 1)


def shutdown_dataloader(dl: DataLoader) -> None:
    """Release DataLoader worker processes and file descriptors."""
    if dl is None:
        return
    iterator = getattr(dl, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
        dl._iterator = None
    if hasattr(dl, "_workers"):
        dl._workers = None


def get_dataloader(ds: Dataset, batch_size: int = 8, isDistributed: bool = False) -> DataLoader:
    num_workers = _num_dataloader_workers()
    common = dict(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=set_seed,
    )
    if num_workers > 0:
        common["num_workers"] = num_workers
        common["prefetch_factor"] = 2

    if isDistributed:
        return DataLoader(
            **common,
            sampler=DistributedSampler(ds, shuffle=False),
        )
    return DataLoader(**common, shuffle=False)


def load_base_cfg(config_filename: str = "config"):
    global _BASE_CFG
    if _BASE_CFG is None:
        yaml_cfg = OmegaConf.load(CONFIG_PATH)
        structured_cfg = OmegaConf.structured(InfConfig)
        _BASE_CFG = OmegaConf.merge(structured_cfg, yaml_cfg)
    return OmegaConf.create(OmegaConf.to_container(_BASE_CFG, resolve=True))


def build_job_cfg(
    config_filename: str,
    dataset_filename: str,
    model_path: str,
    batch_size: int,
):
    cfg = load_base_cfg(config_filename)
    OmegaConf.update(cfg, "file.inference", dataset_filename)
    OmegaConf.update(cfg, "file.model", model_path)
    OmegaConf.update(cfg, "dir.image", IMAGE_DIR)
    OmegaConf.update(cfg, "data.batch_size", batch_size)

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_state_dir, exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)
    return cfg


def inference(
    config_filename: str,
    dataset_filename: str,
    model_path: str,
    batch_size: int,
    isDistributed: bool = False,
):
    global _LOG
    cfg = build_job_cfg(config_filename, dataset_filename, model_path, batch_size)

    if _LOG is None:
        _LOG = base_logger(f"{OUTPUT_DIR}/inference_{config_filename}.log")

    if isDistributed:
        ddp_setup()

    if not os.path.exists(cfg.file.model):
        if isDistributed:
            dist.destroy_process_group()
        return None

    cds = InfCustomDataset(
        record_file=cfg.file.inference,
        img_dir=cfg.dir.image,
        img_map=cfg.data.interpolator_images_map,
    )
    ds = cds._get_inference_dl()
    dl = get_dataloader(ds=ds, batch_size=cfg.data.batch_size, isDistributed=isDistributed)

    try:
        model, device = InferenceLib.get_or_load_model(
            cfg, _LOG, cfg.file.model, _MODEL_CACHE, isDistributed=isDistributed
        )
        model_metrics, linear_metrics, of_metrics = InferenceLib.evaluate_with_baselines(
            model,
            device,
            dl,
            linear_interpolation,
            of_interpolation,
            show_progress=True,
        )
        return InferenceLib.metrics_tuple(model_metrics, linear_metrics, of_metrics)
    finally:
        shutdown_dataloader(dl)
        del ds, dl, cds
        if isDistributed:
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def linear_interpolation(x0, x2, t=0.5):
    return (1.0 - t) * x0 + t * x2


def round_metric_values(row):
    rounded = {}
    for key, value in row.items():
        if key in RESULT_ID_COLS:
            rounded[key] = value
        else:
            rounded[key] = round(float(value), METRIC_PRECISION)
    return rounded


def append_result_row(row):
    df_row = pd.DataFrame([round_metric_values(row)])
    write_header = not os.path.exists(OUTPUT_FILE)
    float_fmt = f"%.{METRIC_PRECISION}f"
    df_row.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=write_header,
        float_format=float_fmt,
    )
    # print(f"Saved result to {OUTPUT_FILE}")


def save_summary():
    if not os.path.exists(OUTPUT_FILE):
        return
    df = pd.read_csv(OUTPUT_FILE)
    metric_cols = [c for c in df.columns if c not in RESULT_ID_COLS]
    grouped_df = df.groupby(["resolution", "patch_size"])[
        metric_cols].mean().reset_index()
    grouped_df[metric_cols] = grouped_df[metric_cols].round(METRIC_PRECISION)
    float_fmt = f"%.{METRIC_PRECISION}f"
    grouped_df.to_csv(SUMMARY_FILE, index=False, float_format=float_fmt)
    # print(f"Updated summary at {SUMMARY_FILE}")


def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    if os.path.exists(SUMMARY_FILE):
        os.remove(SUMMARY_FILE)
    load_base_cfg("config")

    saved_count = 0
    for resolution in RESOLUTIONS:
        for patch in PATCHES:
            for batch_size in BATCHES:
                for organism, samples in TEST_DATASET.items():
                    for sample, subsamples in samples.items():
                        for subsample, content in subsamples.items():
                            for triplet in content["triplets"]:
                                uuid = triplet[1]
                                for chromosome in CHROMOSOMES[organism]:
                                    ds_dict_filename = (
                                        f"{DICT_DIR}/test_{resolution}_{patch}_"
                                        f"{organism}_{uuid}_{chromosome}.txt"
                                    )
                                    print(f"Processing {ds_dict_filename}")
                                    if resolution == 25000:
                                        res = "25k"
                                    elif resolution == 10000:
                                        res = "10k"
                                    elif resolution == 5000:
                                        res = "5k"
                                    else:
                                        print(
                                            f"[WARN] Unsupported resolution: {resolution}")
                                        continue

                                    model_path = f"{MODEL_DIR}/config_a1_{res}_p{patch}_b{batch_size}/hicinterpolate_{patch}_p{patch}_b{batch_size}.pt"
                                    if not os.path.exists(ds_dict_filename) or not os.path.exists(model_path):
                                        print(
                                            f"[WARN] Missing input file or model: "
                                            f"{ds_dict_filename} or {model_path}")
                                        continue

                                    result = inference(
                                        config_filename="config",
                                        dataset_filename=ds_dict_filename,
                                        model_path=model_path,
                                        batch_size=batch_size,
                                        isDistributed=False,
                                    )
                                    if result is None:
                                        continue

                                    (
                                        psnr, ssim, genome_disco, hicrep, lpips,
                                        linear_psnr, linear_ssim, linear_genome_disco,
                                        linear_hicrep, linear_lpips,
                                        of_psnr, of_ssim, of_genome_disco,
                                        of_hicrep, of_lpips,
                                    ) = result

                                    row = {
                                        "resolution": resolution,
                                        "patch_size": patch,
                                        "organism": organism,
                                        "sample": sample,
                                        "subsample": subsample,
                                        "frame": triplet[1],
                                        "chromosome": chromosome,
                                        "psnr_HiCInterpolate": psnr,
                                        "psnr_Linear": linear_psnr,
                                        "psnr_OF": of_psnr,
                                        "ssim_HiCInterpolate": ssim,
                                        "ssim_Linear": linear_ssim,
                                        "ssim_OF": of_ssim,
                                        "genome_disco_HiCInterpolate": genome_disco,
                                        "genome_disco_Linear": linear_genome_disco,
                                        "genome_disco_OF": of_genome_disco,
                                        "hicrep_HiCInterpolate": hicrep,
                                        "hicrep_Linear": linear_hicrep,
                                        "hicrep_OF": of_hicrep,
                                        "lpips_HiCInterpolate": lpips,
                                        "lpips_Linear": linear_lpips,
                                        "lpips_OF": of_lpips,
                                    }
                                    append_result_row(row)
                                    save_summary()
                                    saved_count += 1

    if saved_count == 0:
        print("[WARN] No results were generated to save.")
        return

    print(f"Finished {saved_count} jobs. Results in {OUTPUT_FILE}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        run_inference()
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
