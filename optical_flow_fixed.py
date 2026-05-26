import os
from statistics import mean

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import convolve

from src.metric.eval_metrics import (
    get_genome_disco,
    get_hicrep,
    get_lpips,
    get_psnr,
    get_ssim,
)


DATAPATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/triplets_dataset"
OUTPUT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/output"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "comparison_optical_flow_fixed.csv")

RESOLUTIONS = [25000, 10000, 5000]
PATCHES = [64, 128]
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
                    ],
                ]
            }
        }
    },
}


def load_and_normalize(path):
    img = np.load(path).astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val <= min_val:
        print(
            f"[WARN] Image at {path} has no variation (min={min_val}, max={max_val}). Returning zeros.")
        return np.zeros_like(img, dtype=np.float32)
    mat = ((img - min_val) / (max_val - min_val)).astype(np.float32)
    mat[mat <= 0] = 1e-9
    return mat


def warp_from_endpoint(img, u, v, time_step=0.5):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x - time_step * u).astype(np.float32)
    map_y = (y - time_step * v).astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def dense_lucas_kanade(img1, img2, win_size=15, det_threshold=1e-5):
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

    det = Sxx * Syy - Sxy**2
    mask = det > det_threshold
    u = np.zeros_like(img1, dtype=np.float32)
    v = np.zeros_like(img1, dtype=np.float32)
    u[mask] = (-Syy[mask] * Sxt[mask] + Sxy[mask] * Syt[mask]) / det[mask]
    v[mask] = (Sxy[mask] * Sxt[mask] - Sxx[mask] * Syt[mask]) / det[mask]
    return u, v


def horn_schunck(img1, img2, alpha=1.0, iterations=100):
    u = np.zeros_like(img1, dtype=np.float32)
    v = np.zeros_like(img1, dtype=np.float32)

    Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
    It = img2 - img1

    kernel_avg = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6, 0, 1 / 6],
            [1 / 12, 1 / 6, 1 / 12],
        ],
        dtype=np.float32,
    )

    for _ in range(iterations):
        u_avg = convolve(u, kernel_avg, mode="reflect")
        v_avg = convolve(v, kernel_avg, mode="reflect")
        der = (Ix * u_avg + Iy * v_avg + It) / (alpha**2 + Ix**2 + Iy**2)
        u = u_avg - Ix * der
        v = v_avg - Iy * der

    return u.astype(np.float32), v.astype(np.float32)


def farneback_flow(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(
        img1,
        img2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow[..., 0], flow[..., 1]


def bidirectional_middle_frame(frame1, frame3, flow_fn):
    u13, v13 = flow_fn(frame1, frame3)
    u31, v31 = flow_fn(frame3, frame1)

    mid_from_1 = warp_from_endpoint(frame1, u13, v13, time_step=0.5)
    mid_from_3 = warp_from_endpoint(frame3, u31, v31, time_step=0.5)
    return np.clip(0.5 * mid_from_1 + 0.5 * mid_from_3, 0.0, 1.0).astype(np.float32)


def safe_metric(metric_name, metric_fn, pred_tensor, true_tensor, metadata):
    try:
        value = metric_fn(pred_tensor, true_tensor).item()
    except (ZeroDivisionError, FloatingPointError, ValueError) as exc:
        print(
            "[WARN] "
            f"{metric_name} failed for "
            f"resolution={metadata['resolution']}, patch={metadata['patch_size']}, "
            f"organism={metadata['organism']}, uuid={metadata['uuid']}, "
            f"chromosome={metadata['chromosome']}: {exc}"
        )
        return np.nan

    if not np.isfinite(value):
        print(
            "[WARN] "
            f"{metric_name} returned non-finite value for "
            f"resolution={metadata['resolution']}, patch={metadata['patch_size']}, "
            f"organism={metadata['organism']}, uuid={metadata['uuid']}, "
            f"chromosome={metadata['chromosome']}: {value}"
        )
        return np.nan

    return value


def append_metrics(rows, metadata, pred_lk, pred_hs, pred_fb, true_frame):
    true_tensor = torch.from_numpy(true_frame).unsqueeze(0).unsqueeze(0)
    pred_lk_tensor = torch.from_numpy(pred_lk).unsqueeze(0).unsqueeze(0)
    pred_hs_tensor = torch.from_numpy(pred_hs).unsqueeze(0).unsqueeze(0)
    pred_fb_tensor = torch.from_numpy(pred_fb).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        metric_inputs = {
            "lk": pred_lk_tensor,
            "hs": pred_hs_tensor,
            "fb": pred_fb_tensor,
        }
        metric_fns = {
            "psnr": get_psnr,
            "ssim": get_ssim,
            "genome_disco": get_genome_disco,
            "hicrep": get_hicrep,
            "lpips": get_lpips,
        }

        for suffix, pred_tensor in metric_inputs.items():
            for metric_name, metric_fn in metric_fns.items():
                rows[f"{metric_name}_{suffix}"].append(
                    safe_metric(
                        f"{metric_name}_{suffix}",
                        metric_fn,
                        pred_tensor,
                        true_tensor,
                        metadata,
                    )
                )


def empty_metric_lists():
    return {
        "psnr_lk": [],
        "psnr_hs": [],
        "psnr_fb": [],
        "ssim_lk": [],
        "ssim_hs": [],
        "ssim_fb": [],
        "genome_disco_lk": [],
        "genome_disco_hs": [],
        "genome_disco_fb": [],
        "hicrep_lk": [],
        "hicrep_hs": [],
        "hicrep_fb": [],
        "lpips_lk": [],
        "lpips_hs": [],
        "lpips_fb": [],
    }


def build_result_row(metadata, metrics):
    row = dict(metadata)
    for key, values in metrics.items():
        finite_values = [value for value in values if np.isfinite(value)]
        row[key] = mean(finite_values) if finite_values else np.nan
    return row


def prepare_triplets():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    first_write = True

    for resolution in RESOLUTIONS:
        for patch in PATCHES:
            for organism, samples in DATASET.items():
                for sample, subsamples in samples.items():
                    for subsample, content in subsamples.items():
                        for triplet in content["triplets"]:
                            uuid = "_".join(triplet)
                            for chromosome in CHROMOSOMES[organism]:
                                input_file = (
                                    f"{DATAPATH}/test/test_{resolution}_{patch}_"
                                    f"{organism}_{uuid}_{chromosome}.txt"
                                )
                                print(f"Processing {input_file}")

                                if not os.path.exists(input_file):
                                    print(
                                        f"[WARN] Missing input file: {input_file}")
                                    continue

                                metrics = empty_metric_lists()
                                metadata = {
                                    "resolution": resolution,
                                    "patch_size": patch,
                                    "organism": organism,
                                    "sample": sample,
                                    "subsample": subsample,
                                    "uuid": uuid,
                                    "frame": triplet[1],
                                    "chromosome": chromosome,
                                }

                                with open(input_file, "r", encoding="utf-8") as infile:
                                    for line in infile:
                                        line = line.strip()
                                        if not line:
                                            continue

                                        frame1 = load_and_normalize(
                                            f"{DATAPATH}/{line}/img1.npy")
                                        true_frame = load_and_normalize(
                                            f"{DATAPATH}/{line}/img2.npy")
                                        frame3 = load_and_normalize(
                                            f"{DATAPATH}/{line}/img3.npy")

                                        pred_lk = bidirectional_middle_frame(
                                            frame1,
                                            frame3,
                                            lambda a, b: dense_lucas_kanade(
                                                a, b, win_size=15),
                                        )
                                        pred_hs = bidirectional_middle_frame(
                                            frame1,
                                            frame3,
                                            lambda a, b: horn_schunck(
                                                a, b, alpha=1.0, iterations=100),
                                        )
                                        pred_fb = bidirectional_middle_frame(
                                            frame1,
                                            frame3,
                                            farneback_flow,
                                        )

                                        append_metrics(
                                            metrics,
                                            metadata,
                                            pred_lk,
                                            pred_hs,
                                            pred_fb,
                                            true_frame,
                                        )

                                result_row = pd.DataFrame(
                                    [build_result_row(metadata, metrics)])
                                result_row.to_csv(
                                    OUTPUT_FILE,
                                    mode="w" if first_write else "a",
                                    header=first_write,
                                    index=False,
                                )
                                first_write = False


if __name__ == "__main__":
    try:
        prepare_triplets()
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
