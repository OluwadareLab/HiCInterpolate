import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import convolve
import torch
from statistics import mean
from src.metric.eval_metrics import get_psnr, get_ssim, get_genome_disco, get_hicrep, get_lpips

DATAPATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset"
OUTPUT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/output"

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128, 256, 512]
CHROMOSOMES = {
    "human": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y"],
    "mouse": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "X", "Y"]
}

DATASET = {
    "human": {
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
        },

        "hct116": {
            "2": {
                "triplets":
                [
                    ["4DNFI5IZNXIO_hct116_2_no_transcription_360m_20m",
                     "4DNFIZK7W8GZ_hct116_2_no_transcription_360m_40m",
                     "4DNFISRP84FE_hct116_2_no_transcription_360m_60m"],

                    ["4DNFII16KXA7_hct116_2_no_transcription_60m_20m",
                     "4DNFIMIMLMD3_hct116_2_no_transcription_60m_40m",
                     "4DNFI2LY7B73_hct116_2_no_transcription_60m_60m"],

                    ["4DNFITUPI4HA_hct116_2_no_atp_120m_20m",
                     "4DNFIM7Q2FQQ_hct116_2_no_atp_120m_40m",
                     "4DNFISATK9PF_hct116_2_no_atp_120m_60m"],

                    ["4DNFIVC8OQPG_hct116_2_no_atp_30m_20m",
                     "4DNFI44JLUSL_hct116_2_no_atp_30m_40m",
                     "4DNFIBED48O1_hct116_2_no_atp_30m_60m"],

                    ["4DNFIDD9IF9T_hct116_2_no_replication_20m",
                     "4DNFIQWWATGK_hct116_2_no_replication_40m",
                     "4DNFI3NTD7B3_hct116_2_no_replication_60m"]
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


def load_and_normalize(path):
    img = np.load(path).astype(np.float32)

    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)
    return img


def warp_frame(img, u, v, time_step=0.5):
    """Warps an image along the flow vectors (u, v) by a fractional time_step."""
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (x + time_step * u).astype(np.float32)
    map_y = (y + time_step * v).astype(np.float32)

    warped_img = cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped_img


def dense_lucas_kanade(img1, img2, win_size=15):
    Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
    It = img2 - img1

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    kernel = np.ones((win_size, win_size), np.float32)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Sxt = cv2.filter2D(Ixt, -1, kernel)
    Syt = cv2.filter2D(Iyt, -1, kernel)

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    det = Sxx * Syy - Sxy**2
    mask = det > 1e-5

    u[mask] = (-Syy[mask] * Sxt[mask] + Sxy[mask] * Syt[mask]) / det[mask]
    v[mask] = (Sxy[mask] * Sxt[mask] - Sxx[mask] * Syt[mask]) / det[mask]

    return u, v


def horn_schunck(img1, img2, alpha=1.0, iterations=100):
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
    It = img2 - img1

    kernel_avg = np.array([[1/12, 1/6, 1/12],
                           [1/6,    0, 1/6],
                           [1/12, 1/6, 1/12]], dtype=np.float32)

    for _ in range(iterations):
        u_avg = convolve(u, kernel_avg, mode='reflect')
        v_avg = convolve(v, kernel_avg, mode='reflect')

        der = (Ix * u_avg + Iy * v_avg + It) / (alpha**2 + Ix**2 + Iy**2)

        u = u_avg - Ix * der
        v = v_avg - Iy * der

    return u, v


i = 0


def prepare_triplates():
    for resolution in RESOLUTIONS:
        for patch in PATCHES:
            for organism, samples in DATASET.items():
                for sample, subsamples in samples.items():
                    for subsample, content in subsamples.items():
                        for triplet in content["triplets"]:
                            uuid = triplet[0] + "_" + \
                                triplet[1] + "_" + triplet[2]
                            for chromosome in CHROMOSOMES[organism]:
                                psnrs_lk = []
                                psnrs_hs = []
                                psnrs_fb = []

                                ssims_lk = []
                                ssims_hs = []
                                ssims_fb = []

                                genome_discos_lk = []
                                genome_discos_hs = []
                                genome_discos_fb = []

                                hicreps_lk = []
                                hicreps_hs = []
                                hicreps_fb = []

                                lpipss_lk = []
                                lpipss_hs = []
                                lpipss_fb = []

                                input_file = f"{DATAPATH}/test/test_{resolution}_{patch}_{organism}_{uuid}_{chromosome}.txt"
                                print(f"Processing {input_file}")
                                with open(input_file, "r") as infile:
                                    for line in infile:
                                        line = line.strip()

                                        frame1 = load_and_normalize(
                                            f"{DATAPATH}/{line}/img1.npy")
                                        true_frame = load_and_normalize(
                                            f"{DATAPATH}/{line}/img2.npy")
                                        frame2 = load_and_normalize(
                                            f"{DATAPATH}/{line}/img3.npy")

                                        img1_cv = frame1.astype(np.float32)
                                        img2_cv = frame2.astype(np.float32)

                                        flow_fb = cv2.calcOpticalFlowFarneback(
                                            img1_cv, img2_cv, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                        u_fb, v_fb = flow_fb[...,
                                                             0], flow_fb[..., 1]
                                        pred_fb = warp_frame(
                                            img1_cv, u_fb, v_fb, time_step=0.5)

                                        u_lk, v_lk = dense_lucas_kanade(
                                            img1_cv, img2_cv, win_size=15)
                                        pred_lk = warp_frame(
                                            img1_cv, u_lk, v_lk, time_step=0.5)

                                        u_hs, v_hs = horn_schunck(
                                            img1_cv, img2_cv, alpha=1.0, iterations=100)
                                        pred_hs = warp_frame(
                                            img1_cv, u_hs, v_hs, time_step=0.5)

                                        true_tensor = torch.from_numpy(
                                            true_frame).unsqueeze(0).unsqueeze(0)

                                        pred_lk_tensor = torch.from_numpy(
                                            pred_lk).unsqueeze(0).unsqueeze(0)
                                        pred_hs_tensor = torch.from_numpy(
                                            pred_hs).unsqueeze(0).unsqueeze(0)
                                        pred_fb_tensor = torch.from_numpy(
                                            pred_fb).unsqueeze(0).unsqueeze(0)

                                        psnrs_lk.append(
                                            get_psnr(pred_lk_tensor, true_tensor).item())
                                        psnrs_hs.append(
                                            get_psnr(pred_hs_tensor, true_tensor).item())
                                        psnrs_fb.append(
                                            get_psnr(pred_fb_tensor, true_tensor).item())

                                        ssims_lk.append(
                                            get_ssim(pred_lk_tensor, true_tensor).item())
                                        ssims_hs.append(
                                            get_ssim(pred_hs_tensor, true_tensor).item())
                                        ssims_fb.append(
                                            get_ssim(pred_fb_tensor, true_tensor).item())

                                        genome_discos_lk.append(get_genome_disco(
                                            pred_lk_tensor, true_tensor).item())
                                        genome_discos_hs.append(get_genome_disco(
                                            pred_hs_tensor, true_tensor).item())
                                        genome_discos_fb.append(get_genome_disco(
                                            pred_fb_tensor, true_tensor).item())

                                        hicreps_lk.append(get_hicrep(
                                            pred_lk_tensor, true_tensor).item())
                                        hicreps_hs.append(get_hicrep(
                                            pred_hs_tensor, true_tensor).item())
                                        hicreps_fb.append(get_hicrep(
                                            pred_fb_tensor, true_tensor).item())

                                        lpipss_lk.append(
                                            get_lpips(pred_lk_tensor, true_tensor).item())
                                        lpipss_hs.append(
                                            get_lpips(pred_hs_tensor, true_tensor).item())
                                        lpipss_fb.append(
                                            get_lpips(pred_fb_tensor, true_tensor).item())

                                    infile.close()

                                row = pd.DataFrame([{
                                    "resolution": resolution,
                                    "patch_size": patch,
                                    "organism": organism,
                                    "sample": sample,
                                    "subsample": subsample,
                                    "uuid": uuid,
                                    "frame": triplet[1],
                                    "chromosome": chromosome,
                                    "psnr_lk": mean(psnrs_lk),
                                    "psnr_hs": mean(psnrs_hs),
                                    "psnr_fb": mean(psnrs_fb),
                                    "ssim_lk": mean(ssims_lk),
                                    "ssim_hs": mean(ssims_hs),
                                    "ssim_fb": mean(ssims_fb),
                                    "genome_disco_lk": mean(genome_discos_lk),
                                    "genome_disco_hs": mean(genome_discos_hs),
                                    "genome_disco_fb": mean(genome_discos_fb),
                                    "hicrep_lk": mean(hicreps_lk),
                                    "hicrep_hs": mean(hicreps_hs),
                                    "hicrep_fb": mean(hicreps_fb),
                                    "lpips_lk": mean(lpipss_lk),
                                    "lpips_hs": mean(lpipss_hs),
                                    "lpips_fb": mean(lpipss_fb)
                                }])

                                row.to_csv(
                                    f"{OUTPUT_PATH}/comparison.csv", mode="a",
                                    header=(i == 0),
                                    index=False)
                                i += 1


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
