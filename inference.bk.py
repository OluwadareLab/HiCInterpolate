import os
import numpy as np
import random
import csv
import sys
import logging
import torch
import torch.distributed as dist
import argparse
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from src.metric import eval_metrics as eval_metric
from src.misc import plots as plot
from src.inference import InfConfig, InfCustomDataset
from src import InferenceLib
from scipy.ndimage import gaussian_filter as sp_gf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_PATH = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
RESOLUTIONS = [10000]

ORGANISMS = [
    "human"
]

SAMPLES = [
    [
        "dmso_control",
        "dtag_v1",
        "hela_s3_r1",
        "hct116_2"
    ]
]

SUBSAMPLES = [
    [
        [
            "control"
        ],
        [
            "v1"
        ],
        [
            "r1"
        ],
        [
            "noatp30m",
            "noatp120m",
            "notranscription60m",
            "notranscription360m"
        ]
    ]
]

FILENAME_LIST = [
    [
        [
            [
                "4DNFIP9EJSOM_dmso_0m",
                "4DNFI7T93SHL_dmso_30m",
                "4DNFICF2Z2TG_dmso_60m"
            ]
        ],
        [
            [
                "4DNFI5EAPQTI_dtag_v1_0m",
                "4DNFIY1TCVLX_dtag_v1_30m",
                "4DNFIXWT5U42_dtag_v1_60m"
            ]
        ],
        [
            [
                "4DNFIZZ77KD2_hela_s3_r1_30m",
                "4DNFIOLO226X_hela_s3_r1_60m",
                "4DNFIJMS2ODT_hela_s3_r1_90m"
            ]
        ],
        [
            [
                "4DNFIVC8OQPG_hct116_2_noatp30m_20m",
                "4DNFI44JLUSL_hct116_2_noatp30m_40m",
                "4DNFIBED48O1_hct116_2_noatp30m_60m"
            ],
            [
                "4DNFITUPI4HA_hct116_2_noatp120m_20m",
                "4DNFIM7Q2FQQ_hct116_2_noatp120m_40m",
                "4DNFISATK9PF_hct116_2_noatp120m_60m"
            ],
            [
                "4DNFII16KXA7_hct116_2_notranscription60m_20m",
                "4DNFIMIMLMD3_hct116_2_notranscription60m_40m",
                "4DNFI2LY7B73_hct116_2_notranscription60m_60m"
            ],
            [
                "4DNFI5IZNXIO_hct116_2_notranscription360m_20m",
                "4DNFIZK7W8GZ_hct116_2_notranscription360m_40m",
                "4DNFISRP84FE_hct116_2_notranscription360m_60m"
            ]
        ]
    ]
]

CHROMOSOMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']


_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


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


def reconstruct_matrix(pred_list, patch_size, h, w):
    n = int(math.ceil(h / patch_size) * patch_size)
    n_patches_per_side = n // patch_size

    reconstructed = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for batch in pred_list:
        # batch may be a torch.Tensor on CPU or a numpy array
        if isinstance(batch, torch.Tensor):
            # expected shapes: (B,1,patch,patch) or (B,patch,patch)
            if batch.dim() == 4:
                arr = batch.squeeze(1).numpy()
            else:
                arr = batch.numpy()
        else:
            arr = np.array(batch)

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        for p in arr:
            row = idx // n_patches_per_side
            col = idx % n_patches_per_side
            reconstructed[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = p
            idx += 1

    pred = reconstructed[:h, :w]
    return pred


def get_norm_mat(matrix, gf: bool = False, log: bool = False, clip: bool = False, percentile: float = CLIPPING_PERCENTILE):
    mat = np.nan_to_num(matrix, nan=_EPSILON, posinf=_EPSILON, neginf=_EPSILON)
    mat[mat <= _EPSILON] = _EPSILON
    if gf:
        mat = sp_gf(mat, 1.0)
    if log:
        mat = np.log1p(mat)
    if clip:
        percentile_val = np.percentile(mat, percentile)
        mat = np.clip(mat, _EPSILON, percentile_val)

    _min = np.min(mat)
    _max = np.max(mat)
    mat = (mat - _min)/(_max - _min)
    return mat


def filter_negative(matrix):
    matrix[matrix <= _EPSILON] = _EPSILON
    return matrix


def log_clip_min_max(matrix, pv=-1, percentile: float = CLIPPING_PERCENTILE):
    matrix = np.nan_to_num(matrix, nan=_EPSILON,
                           posinf=_EPSILON, neginf=_EPSILON)
    matrix[matrix < _EPSILON] = _EPSILON
    log_matrix = np.log1p(matrix)
    percentile_val = np.percentile(log_matrix, percentile)
    clip_matrix = np.clip(log_matrix, _EPSILON, percentile_val)
    norm_matrix = clip_matrix / percentile_val

    # _min = np.min(log_matrix)
    # if max != -1:
    #     _max = max
    # else:
    #     _max = np.max(log_matrix)

    # mat = (log_matrix - _min)/(_max - _min)
    # mat[mat == 0] = _EPSILON

    return norm_matrix, percentile_val


def get_score_matrix(preds, target, device, patch_size, batch_size):
    patches_y = []
    patches_pred = []
    h, w = target.shape
    n_y = get_norm_mat(target, gf=True)
    n_pred = preds

    bin = 0
    while (bin+patch_size <= h and bin+patch_size <= w):
        temp_y = n_y[bin:bin+patch_size, bin:bin+patch_size]
        temp_pred = n_pred[bin:bin+patch_size, bin:bin+patch_size]

        if temp_y.shape == (patch_size, patch_size):
            patches_y.append(temp_y)
            patches_pred.append(temp_pred)
        bin += patch_size

    patches_y = torch.from_numpy(np.stack(patches_y)).to(
        device).unsqueeze(1).float()
    patches_pred = torch.from_numpy(
        np.stack(patches_pred)).to(device).unsqueeze(1).float()

    num_patches = len(patches_y)

    psnr_values = []
    ssim_values = []
    genome_disco_values = []
    hicrep_values = []
    lpips_values = []

    for batch_start in range(0, num_patches, batch_size):
        batch_end = min(batch_start + batch_size, num_patches)

        batch_y = patches_y[batch_start:batch_end]
        batch_pred = patches_pred[batch_start:batch_end]

        psnr_values.append(eval_metric.get_psnr(batch_pred, batch_y))
        ssim_values.append(eval_metric.get_ssim(batch_pred, batch_y))
        genome_disco_values.append(
            eval_metric.get_genome_disco(batch_pred, batch_y))
        hicrep_values.append(eval_metric.get_hicrep(batch_pred, batch_y))
        lpips_values.append(eval_metric.get_lpips(batch_pred, batch_y))

    psnr = np.mean([v.cpu().item() if torch.is_tensor(v)
                   else v for v in psnr_values])
    ssim = np.mean([v.cpu().item() if torch.is_tensor(v)
                   else v for v in ssim_values])
    genome_disco = np.mean([v.cpu().item() if torch.is_tensor(
        v) else v for v in genome_disco_values])
    hicrep = np.mean([v.cpu().item() if torch.is_tensor(v)
                     else v for v in hicrep_values])
    lpips = np.mean([v.cpu().item() if torch.is_tensor(v)
                    else v for v in lpips_values])

    return psnr, ssim, genome_disco, hicrep, lpips


# def get_score_matrix(preds, target, device, patch_size):
#     n_y = get_norm_mat(target)
#     n_pred = get_norm_mat(preds)

#     h, w = target.shape
#     bin = 0

#     psnr_values = []
#     ssim_values = []
#     genome_disco_values = []
#     hicrep_values = []
#     lpips_values = []
#     while(bin+patch_size <= h and bin+patch_size <= w):
#         temp_y = n_y[bin:bin+patch_size, bin:bin+patch_size]
#         temp_pred = n_pred[bin:bin+patch_size, bin:bin+patch_size]
#         patche_y = torch.from_numpy(temp_y).to(device).unsqueeze(0).unsqueeze(0).float()
#         patche_pred = torch.from_numpy(temp_pred).to(device).unsqueeze(0).unsqueeze(0).float()

#         psnr_values.append(eval_metric.get_psnr(patche_pred, patche_y))
#         ssim_values.append(eval_metric.get_ssim(patche_pred, patche_y))
#         genome_disco_values.append(eval_metric.get_genome_disco(patche_pred, patche_y))
#         hicrep_values.append(eval_metric.get_hicrep(patche_pred, patche_y))
#         lpips_values.append(eval_metric.get_lpips(patche_pred, patche_y))

#         bin += patch_size


#     psnr = np.mean([v.cpu().item() if torch.is_tensor(v) else v for v in psnr_values])
#     ssim = np.mean([v.cpu().item() if torch.is_tensor(v) else v for v in ssim_values])
#     genome_disco = np.mean([v.cpu().item() if torch.is_tensor(v) else v for v in genome_disco_values])
#     hicrep = np.mean([v.cpu().item() if torch.is_tensor(v) else v for v in hicrep_values])
#     lpips = np.mean([v.cpu().item() if torch.is_tensor(v) else v for v in lpips_values])

#     return psnr, ssim, genome_disco, hicrep, lpips


def main(config_filename: str, isDistributed: bool = False):
    yaml_cfg = OmegaConf.load(f"./src/inference/{config_filename}.yaml")
    structured_cfg = OmegaConf.structured(InfConfig)
    cfg = OmegaConf.merge(structured_cfg, yaml_cfg)

    output_dir = f"{cfg.dir.output}/{config_filename}"
    model_state_dir = f"{cfg.dir.model_state}/{config_filename}"
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{model_state_dir}", exist_ok=True)
    OmegaConf.update(cfg, "dir.output", output_dir)

    log = base_logger(cfg.file.log)
    if isDistributed:
        ddp_setup()

    batch_size = cfg.data.batch_size
    patch_size = cfg.data.patch
    device = cfg.device
    csv_file = 'scores.csv'
    fieldnames = ['chromosome', 'psnr', 'ssim',
                  'genome_disco', 'hicrep', 'lpips']
    if os.path.exists(cfg.file.model):
        for organism, org_samples, org_subsamples, org_filenames in zip(ORGANISMS, SAMPLES, SUBSAMPLES, FILENAME_LIST):
            for sample, sam_sub_sample, sam_sample_filenames in zip(org_samples, org_subsamples, org_filenames):
                for sub_sample, sample_filenames in zip(sam_sub_sample, sam_sample_filenames):
                    for resolution in RESOLUTIONS:
                        for chromosome in CHROMOSOMES:
                            timeframe_name = "_".join(name.split(
                                '_')[-1] for name in sample_filenames[:3])
                            print(
                                f"Processing {patch_size}/{organism}/{sample}/{sub_sample}/{timeframe_name}/{str(resolution)}/{chromosome}...")
                            data_dict = f"{ROOT_PATH}/inference/kr_gf/{patch_size}/{organism}/{sample}/{sub_sample}/{timeframe_name}/{str(resolution)}/{chromosome}"
                            OmegaConf.update(cfg, "dir.image", data_dict)

                            cds = InfCustomDataset(record_file=cfg.file.inference, img_dir=cfg.dir.image,
                                                   img_map=cfg.data.interpolator_images_map)
                            ds = cds._get_inference_dl()
                            dl = get_dataloader(
                                ds=ds, batch_size=batch_size, isDistributed=isDistributed)
                            inference = InferenceLib.HiCInterpolate(
                                cfg=cfg, log=log, model=cfg.file.model, dl=dl, isDistributed=isDistributed)
                            inference._inference()

                            pred_list = inference._get_prediction()

                            y = np.load(f"{data_dict}/y.npy")
                            h, w = y.shape
                            pred = reconstruct_matrix(
                                pred_list, patch_size, h, w)
                            up_pred = pred * np.max(y)

                            ROOT_OUTPUT = f"{cfg.dir.output}/{patch_size}/{organism}/{sample}/{sub_sample}"
                            os.makedirs(ROOT_OUTPUT, exist_ok=True)
                            MATRIX_DIR = f"{ROOT_OUTPUT}/matrix"
                            os.makedirs(MATRIX_DIR, exist_ok=True)
                            RES_DIR = f"{ROOT_OUTPUT}/{timeframe_name}/{str(resolution)}"
                            os.makedirs(RES_DIR, exist_ok=True)
                            np.save(f"{MATRIX_DIR}/{chromosome}_y.npy", y)
                            np.save(f"{MATRIX_DIR}/{chromosome}_yt.npy", pred)

                            n_y, _max = log_clip_min_max(y)
                            n_pred, _ = log_clip_min_max(up_pred, pv=_max)
                            print(f"Ploating heatmap...")
                            plot.draw_inf_hic_map(
                                y=n_y[3328:3584, 3328:3584], pred=n_pred[3328:3584, 3328:3584], file=f"{RES_DIR}/{chromosome}_hic_map")

                            print(f"Calculating scores...")
                            psnr, ssim, genome_disco, hicrep, lpips = get_score_matrix(
                                pred, y, device, patch_size, batch_size)
                            scores = f"PSNR: {format(psnr, '.4f')}, SSIM: {format(ssim, '.4f')}, GenomeDISCO: {format(genome_disco, '.4f')}, HiCRep: {format(hicrep, '.4f')}, LPIPS: {format(lpips, '.4f')};"

                            csv_file = f"{ROOT_OUTPUT}/scores.csv"
                            if not os.path.exists(csv_file):
                                with open(csv_file, 'w', newline='') as f:
                                    writer = csv.DictWriter(
                                        f, fieldnames=fieldnames)
                                    writer.writeheader()
                            with open(csv_file, 'a', newline='') as f:
                                writer = csv.DictWriter(
                                    f, fieldnames=fieldnames)
                                writer.writerow({
                                    'chromosome': chromosome,
                                    'psnr': psnr,
                                    'ssim': ssim,
                                    'genome_disco': genome_disco,
                                    'hicrep': hicrep,
                                    'lpips': lpips
                                })
                                f.close()

                            print(f"Saved scores for {chromosome}")

                            log.info(f"{scores}")
                            print(f"[INFO] {scores}")

                        if isDistributed:
                            dist.destroy_process_group()


if __name__ == "__main__":
    set_seed(42)
    print("sys.argv:", sys.argv)
    parser = argparse.ArgumentParser(
        description='ap film distributed training job')
    parser.add_argument('-dis', '--distributed', dest="distributed",
                        action='store_true', help='Distributed training (default: False)')
    parser.add_argument('-cfg', '--config', dest="config",  type=str, default="config",
                        help='Configuration filename without extension. This file should be in the configs folder (default: config)')
    args = parser.parse_args()

    # main(args.config, args.distributed)
    main(config_filename="config", isDistributed=False)
