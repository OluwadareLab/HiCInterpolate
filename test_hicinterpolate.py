from tqdm.asyncio import tqdm

from src.metric.eval_metrics import (
    get_genome_disco,
    get_hicrep,
    get_lpips,
    get_psnr,
    get_ssim,
)
import os
from statistics import mean
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import os
import numpy as np
import random
import sys
import logging
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from src.metric import eval_metrics as eval_metric
from src.inference import InfConfig, InfCustomDataset
from src import InferenceLib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate"
DATA_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset"
OUTPUT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/output/test"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "comparison_hicinterpolate.csv")

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128]
BATCHES = [64, 128]
CHROMOSOMES = {
    "human": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y"],
    "mouse": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "X", "Y"],
}

DATASET = {
    "human": {
        "dtag": {
            "v1": {
                "triplets": [
                    [
                        "4DNFI5EAPQTI_dtag_v1_0m",
                        "4DNFIY1TCVLX_dtag_v1_30m",
                        "4DNFIXWT5U42_dtag_v1_60m",
                    ],
                    [
                        "4DNFIY1TCVLX_dtag_v1_30m",
                        "4DNFIXWT5U42_dtag_v1_60m",
                        "4DNFIHTFIMGG_dtag_v1_90m",
                    ],
                    [
                        "4DNFIXWT5U42_dtag_v1_60m",
                        "4DNFIHTFIMGG_dtag_v1_90m",
                        "4DNFIPZCCTV6_dtag_v1_120m",
                    ],
                ]
            }
        },
        "hct116": {
            "2": {
                "triplets": [
                    [
                        "4DNFI5IZNXIO_hct116_2_no_transcription_360m_20m",
                        "4DNFIZK7W8GZ_hct116_2_no_transcription_360m_40m",
                        "4DNFISRP84FE_hct116_2_no_transcription_360m_60m",
                    ],
                    [
                        "4DNFII16KXA7_hct116_2_no_transcription_60m_20m",
                        "4DNFIMIMLMD3_hct116_2_no_transcription_60m_40m",
                        "4DNFI2LY7B73_hct116_2_no_transcription_60m_60m",
                    ],
                    [
                        "4DNFITUPI4HA_hct116_2_no_atp_120m_20m",
                        "4DNFIM7Q2FQQ_hct116_2_no_atp_120m_40m",
                        "4DNFISATK9PF_hct116_2_no_atp_120m_60m",
                    ],
                    [
                        "4DNFIVC8OQPG_hct116_2_no_atp_30m_20m",
                        "4DNFI44JLUSL_hct116_2_no_atp_30m_40m",
                        "4DNFIBED48O1_hct116_2_no_atp_30m_60m",
                    ],
                    [
                        "4DNFIDD9IF9T_hct116_2_no_replication_20m",
                        "4DNFIQWWATGK_hct116_2_no_replication_40m",
                        "4DNFI3NTD7B3_hct116_2_no_replication_60m",
                    ],
                ]
            }
        },
    },
    "mouse": {
        "embryo": {
            "development": {
                "triplets": [
                    [
                        "4DNFIN8F14CS_sperm",
                        "4DNFIVCJKHMN_mii_oocyte",
                        "4DNFI1EYIGOC_zygote",
                    ],
                    [
                        "4DNFIVCJKHMN_mii_oocyte",
                        "4DNFI1EYIGOC_zygote",
                        "4DNFIK4CECUH_early2_cell",
                    ],
                    [
                        "4DNFI1EYIGOC_zygote",
                        "4DNFIK4CECUH_early2_cell",
                        "4DNFICXCFGEI_late2_cell",
                    ],
                    [
                        "4DNFIK4CECUH_early2_cell",
                        "4DNFICXCFGEI_late2_cell",
                        "4DNFIFA89L5B_8cell",
                    ],
                    [
                        "4DNFICXCFGEI_late2_cell",
                        "4DNFIFA89L5B_8cell",
                        "4DNFIK5HY1GP_icm",
                    ],
                    [
                        "4DNFIFA89L5B_8cell",
                        "4DNFIK5HY1GP_icm",
                        "4DNFI5IAH9H1_mes_cell",
                    ]
                ]
            }
        }
    }
}


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
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


def get_dataloader(ds: Dataset, batch_size: int = 8, isDistributed: bool = False) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            worker_init_fn=set_seed,
            sampler=DistributedSampler(ds, shuffle=False)
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=set_seed
        )


def build_result_row(metadata, metrics):
    row = dict(metadata)
    for key, values in metrics.items():
        finite_values = [value for value in values if np.isfinite(value)]
        row[key] = mean(finite_values) if finite_values else np.nan
    return row


def inference(config_filename: str, dataset_filename: str, model_path: str, batch_size: int, isDistributed: bool = False):
    yaml_cfg = OmegaConf.load(f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/src/inference/{config_filename}.yml")
    structured_cfg = OmegaConf.structured(InfConfig)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    OmegaConf.update(cfg, "file.inference", dataset_filename)
    OmegaConf.update(cfg, "file.model", model_path)
    OmegaConf.update(cfg, "dir.image", f"{DATA_PATH}")
    OmegaConf.update(cfg, "data.batch_size", batch_size)

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"

    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)

    log = base_logger(f"{OUTPUT_PATH}/inference_{config_filename}.log")
    if isDistributed:
        ddp_setup()

    batch_size = cfg.data.batch_size
    if os.path.exists(cfg.file.model):
        cds = InfCustomDataset(record_file=cfg.file.inference,
                               img_dir=cfg.dir.image, img_map=cfg.data.interpolator_images_map)
        ds = cds._get_inference_dl()
        dl = get_dataloader(ds=ds, batch_size=batch_size,
                            isDistributed=isDistributed)
        inference = InferenceLib.HiCInterpolate(
            cfg=cfg, log=log, model=cfg.file.model, dl=dl, isDistributed=isDistributed)
        model, device = inference._get_model()
        psnrs = []
        ssims = []
        genome_discos = []
        hicreps = []
        lpips_scores = []
        with torch.no_grad():
            model.eval()
            for _, (x1, x2, x3, time_frame) in enumerate(tqdm(dl)):
                x1 = x1.to(device)
                x3 = x3.to(device)
                time_frame = time_frame.to(device)
                pred = model(x1, x3, time_frame)
                x2 = x2.to(device)
                psnrs.append(get_psnr(pred, x2).item())
                ssims.append(get_ssim(pred, x2).item())
                genome_discos.append(get_genome_disco(pred, x2).item())
                hicreps.append(get_hicrep(pred, x2).item())
                lpips_scores.append(get_lpips(pred, x2).item())
                del x1, x2, x3, time_frame, pred

        psnr_mean = mean(psnrs)
        ssim_mean = mean(ssims)
        genome_disco_mean = mean(genome_discos)
        hicrep_mean = mean(hicreps)
        lpips_mean = mean(lpips_scores)
        return psnr_mean, ssim_mean, genome_disco_mean, hicrep_mean, lpips_mean

    if isDistributed:
        dist.destroy_process_group()


def run_inference():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    first_write = True

    for resolution in RESOLUTIONS:
        for patch in PATCHES:
            for batch_size in BATCHES:
                for organism, samples in DATASET.items():
                    for sample, subsamples in samples.items():
                        for subsample, content in subsamples.items():
                            for triplet in content["triplets"]:
                                uuid = "_".join(triplet)
                                for chromosome in CHROMOSOMES[organism]:
                                    ds_dict_filename = (
                                        f"{DATA_PATH}/test/test_{resolution}_{patch}_"
                                        f"{organism}_{uuid}_{chromosome}.txt"
                                    )
                                    print(f"Processing {ds_dict_filename}")
                                    res = ""
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
                                    model_path = f"{ROOT_PATH}/datasets/output/config_a1_{res}_p{patch}_b{batch_size}/hicinterpolate_{patch}_p{patch}_b{batch_size}.pt"
                                    if not os.path.exists(ds_dict_filename):
                                        print(
                                            f"[WARN] Missing input file: {ds_dict_filename}")
                                        continue


                                    psnr, ssim, genome_disco, hicrep, lpips = inference(
                                        config_filename="config",
                                        dataset_filename=ds_dict_filename,
                                        model_path=model_path,
                                        batch_size=batch_size,
                                        isDistributed=False,
                                    )

                                    # Build row dict
                                    row = {
                                        "resolution": resolution,
                                        "patch_size": patch,
                                        "batch_size": batch_size,
                                        "organism": organism,
                                        "sample": sample,
                                        "subsample": subsample,
                                        "uuid": uuid,
                                        "frame": triplet[1],
                                        "chromosome": chromosome,
                                        "psnr": psnr,
                                        "ssim": ssim,
                                        "genome_disco": genome_disco,
                                        "hicrep": hicrep,
                                        "lpips": lpips,
                                    }

                                    # Save to CSV in append mode
                                    df = pd.DataFrame([row])
                                    if first_write and not os.path.exists(OUTPUT_FILE):
                                        df.to_csv(OUTPUT_FILE, mode="a", header=True, index=False)
                                        first_write = False
                                    else:
                                        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)


if __name__ == "__main__":
    try:
        run_inference()
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
