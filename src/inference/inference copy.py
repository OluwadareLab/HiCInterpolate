import os
import numpy as np
import cooler as cool
import random
import numpy as np
import sys
import os
import logging
import torch
import torch.distributed as dist
import argparse
from src import InferenceLib, CustomDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from configs.config import Config
from src.misc import plots as plot, metrics as metric
from tqdm import tqdm

root_path = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data"
resolutions = [10000]
balance = True
patch_sizes = [64]
cmap = "YlOrRd"

organisms = ["human"]
samples = [["verticular"]]
filename_set = [[[["v_2d_4DNFI1P2EP7L",
                "v_4d_4DNFI7C5YXNX",
                   "v_6d_4DNFI8I2WYXS"]]]]
chromosomes = [21]

# [INFO] PSNR: -20.9029, SSIM: 0.4632, PCC: 0.0000, SCC: 0.0000, GenomeDISCO: -1.0000, NCC: 0.5412, LPIPS: 0.0000;


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


def get_img(chr_mat, r, c, patch, chr_size):
    r_start = min(r, chr_size)
    r_end = min(r+patch, chr_size)
    c_start = min(c, chr_size)
    c_end = min(c+patch, chr_size)
    submatrix = chr_mat[r_start:r_end, c_start:c_end]
    submatrix = submatrix.astype(np.float32)
    return submatrix


def get_patches(mat, patch_size):
    pad_value = 1e-10
    h, w = mat.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    padded = np.pad(mat, ((0, pad_h), (0, pad_w)),
                    mode='constant', constant_values=pad_value)
    patches = []
    for i in range(0, padded.shape[0], patch_size):
        for j in range(0, padded.shape[1], patch_size):
            patch = padded[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    patches = np.stack(patches)

    return padded, patches


def generate_patch(mat_0, mat_y, mat_1, chr_size, organism, sample, resolution, chromosome, sub_sample, model):
    time_ = 0.5
    for patch in patch_sizes:
        row, col = mat_y.shape
        recon = torch.zeros((row, col), dtype=torch.float32)

        time_frame = torch.tensor(
            [[time_]], dtype=torch.float32)
        padded_mat_0, patches_0 = get_patches(
            mat=mat_0, patch_size=patch)
        padded_mat_1, patches_1 = get_patches(
            mat=mat_1, patch_size=patch)

        pred_patches = []
        print(f"[INFO][CUDA] ==== prediction started ====")
        for patch_0, patch_1 in tqdm(zip(patches_0, patches_1), total=len(patches_0), desc="Predicting patches"):
            p0 = torch.from_numpy(patch_0).unsqueeze(0).unsqueeze(0).float()
            p1 = torch.from_numpy(patch_1).unsqueeze(0).unsqueeze(0).float()
            batch = [p0, p1, time_frame]
            model._predict(batch)
            pred = model._get_pred()
            pred_patches.append(pred.squeeze(
                0).squeeze(0).detach().cpu().numpy())
        print(f"[INFO][CUDA] ==== prediction end ====")

        mat_0 = torch.from_numpy(mat_0).to(
            "cuda").unsqueeze(0).unsqueeze(0).float()
        mat_y = torch.from_numpy(mat_y).to(
            "cuda").unsqueeze(0).unsqueeze(0).float()
        mat_1 = torch.from_numpy(mat_1).to(
            "cuda").unsqueeze(0).unsqueeze(0).float()
        if isinstance(padded_mat_0, np.ndarray):
            padded_mat_0 = torch.from_numpy(padded_mat_0)
        pred_padded = torch.zeros_like(padded_mat_0)

        num_patches_h = padded_mat_0.shape[0] // patch
        num_patches_w = padded_mat_0.shape[1] // patch
        i = 0
        for r in range(num_patches_h):
            for c in range(num_patches_w):
                pred_patch = torch.from_numpy(
                    pred_patches[i]).to(pred_padded.dtype)
                pred_padded[
                    r * patch:(r + 1) * patch,
                    c * patch:(c + 1) * patch
                ] = pred_patch
                i += 1

        pred_original = pred_padded[:row, :col]
        pred_original_np = pred_original.to(
            "cuda").unsqueeze(0).unsqueeze(0).float()

        print(f"Pred: {pred_original_np.shape}, Original: {mat_y.shape}")

        print(f"Ploating heatmap...")
        plot.draw_hic_map(num_examples=1, x0=mat_0,
                          y=mat_y, pred=pred_original_np, x1=mat_1, file="hicinterpolate_64_inference_chr_1_10000.png")
        print(f"Heatmap saved...")
        del mat_0, mat_1, time_frame

        print(f"Calculating PSNR...")
        psnr = metric.get_psnr(pred_original_np, mat_y)
        print(f"PSNR: {psnr}")
        print(f"Calculating SSIM...")
        ssim = metric.get_ssim(pred_original_np, mat_y)
        print(f"SSIM: {ssim}")
        pcc = 0
        scc = 0
        print(f"Calculating GenomeDISCO...")
        genome_disco = metric.get_avg_genome_disco(pred_original_np, mat_y)
        print(f"GenomeDISCO: {genome_disco}")
        print(f"Calculating NCC...")
        ncc = metric.get_ncc(pred_original_np, mat_y)
        print(f"NCC: {ncc}")
        # print(f"Creating LPIPS model...")
        # lpips_model = metric.LPIPSLoss(device="cpu")
        # pred_original_t = pred_original.to("cpu", dtype=torch.float32)
        # mat_y_t = torch.from_numpy(mat_y).to("cpu", dtype=torch.float32)
        # print("LPIPS model created...")
        # print("Calculating LPIPS...")
        lpips = 0
        # lpips = lpips_model(pred_original_t, mat_y_t)

        # print(f"LPIPS: {lpips}")
        lpips = 0
        print(
            f"[INFO] PSNR: {format(psnr, '.4f')}, SSIM: {format(ssim, '.4f')}, PCC: {format(pcc, '.4f')}, SCC: {format(scc, '.4f')}, GenomeDISCO: {format(genome_disco, '.4f')}, NCC: {format(ncc, '.4f')}, LPIPS: {format(lpips, '.4f')};")


CONFIG_FILE = f"config_64_set_1_kr_w_rand_AdamW_cosin"
IS_DISTRIBUTED = False
if __name__ == "__main__":
    try:
        yaml_cfg = OmegaConf.load(f"./configs/{CONFIG_FILE}.yaml")
        structured_cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(structured_cfg, yaml_cfg)
        output_dir = f"{cfg.dir.output}/{CONFIG_FILE}"
        model_state_dir = f"{cfg.dir.model_state}/{CONFIG_FILE}"
        os.makedirs(f"{output_dir}", exist_ok=True)
        os.makedirs(f"{model_state_dir}", exist_ok=True)
        OmegaConf.update(cfg, "dir.output", output_dir)
        OmegaConf.update(cfg, "dir.model_state", model_state_dir)

        log = base_logger(cfg.file.log)
        if IS_DISTRIBUTED:
            ddp_setup()
        batch_size = cfg.data.batch_size
        log = base_logger(cfg.file.log)
        hicinterpolate = InferenceLib.HiCInterpolation(cfg, log=log)

        for organism, org_samples, org_filenames in zip(organisms, samples, filename_set):
            for sample, sample_filenames in zip(org_samples, org_filenames):
                for resolution in resolutions:
                    for filenames in sample_filenames:
                        cool_0 = cool.Cooler(
                            f"{root_path}/{organism}/{sample}/{filenames[0]}_{resolution}_kr.cool")
                        cool_y = cool.Cooler(
                            f"{root_path}/{organism}/{sample}/{filenames[1]}_{resolution}_kr.cool")
                        cool_1 = cool.Cooler(
                            f"{root_path}/{organism}/{sample}/{filenames[2]}_{resolution}_kr.cool")

                        sub_sample = "_".join(name.split(
                            '_')[-1] for name in filenames[:3])

                        for chromosome in chromosomes:
                            chr_size = cool_0.chromsizes[f"{chromosome}"]
                            fetch = f"{chromosome}:{0}-{chr_size}"
                            chr_mat_0 = cool_0.matrix(
                                balance=balance).fetch(fetch)
                            chr_mat_y = cool_y.matrix(
                                balance=balance).fetch(fetch)
                            chr_mat_1 = cool_1.matrix(
                                balance=balance).fetch(fetch)
                            generate_patch(chr_mat_0, chr_mat_y, chr_mat_1, chr_size,
                                           organism, sample, resolution, chromosome, sub_sample, hicinterpolate)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
