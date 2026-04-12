import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.pyplot as plt
import cooler as cool
import numpy as np
import os


ROOT_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets"
OUTPUT_ROOT_PATH = f"{ROOT_PATH}/triplates/"

RESOLUTIONS = [25000, 10000, 5000]
BALANCE_COOL = False
PATCHES = [64, 128, 256, 512]
CMAP_ = mcolors.LinearSegmentedColormap.from_list(
    "juicebox",
    [
        "#fee8c8",
        "#fdbb84",
        "#e34a33",
        "#b30000"
    ],
    N=256
)
_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99
PATCH_OVERLAP_RATIO = 0.2

COUTER = {
    5000: {64: 0, 128: 0, 256: 0, 512: 0},
    10000: {64: 0, 128: 0, 256: 0, 512: 0},
    25000: {64: 0, 128: 0, 256: 0, 512: 0}
}


DATA = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    [
                        "4DNFI7T93SHL_dmso_control_30m",
                        "4DNFICF2Z2TG_dmso_control_60m",
                        "4DNFILL624WG_dmso_control_90m",
                    ]
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
                     "4DNFIAUMRM2S_hct116_1_60m"
                     ]
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
                    ["4DNFIWDOOBVE_hela_s3_r1_165m",
                     "4DNFIDT9EB5M_hela_s3_r1_180m",
                     "4DNFIX2VUNV8_hela_s3_r1_195m"]
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
                    ["4DNFIB6PJFJ3_hela_s3_r3_90m",
                     "4DNFIX97731O_hela_s3_r3_105m",
                     "4DNFIYQYZOTO_hela_s3_r3_120m"]
                ]
            },
        }
    }
}


plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 600


def nan_to_eps(matrix):
    return np.nan_to_num(matrix, nan=_EPSILON, posinf=_EPSILON, neginf=_EPSILON)


def log1p(matrix):
    return np.log1p(matrix)


def clip(matrix):
    return np.clip(matrix, _EPSILON, np.percentile(matrix, CLIPPING_PERCENTILE))


def min_max_norm(matrix):
    _min = np.min(matrix)
    _max = np.max(matrix)
    mm_matrix = (matrix - _min) / (_max - _min) if _max > _min else matrix * 0
    mm_matrix[mm_matrix == 0] = _EPSILON
    return mm_matrix


def generate_patch(mat_0, mat_y, mat_1, organism, sample, subsample, resolution, chromosome, output_root_path, uuid):
    data_plot_path = f"{OUTPUT_ROOT_PATH}/data_plots"
    os.makedirs(data_plot_path, exist_ok=True)
    import random
    for patch in PATCHES:
        mats = [mat_0, mat_y, mat_1]
        mat_labels = ["x0", "y", "x1"]
        row_titles = ["Raw", "MinMax", "Log1p", "Clip"]
        col_titles = mat_labels
        # Randomly choose a valid start index for the patch
        mat_shape = mats[0].shape
        max_start = mat_shape[0] - patch
        if max_start < 0:
            continue  # skip if patch is too large for matrix
        start = random.randint(0, max_start//patch)
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        for j, mat in enumerate(mats):
            patch_slice = mat[start:start+patch, start:start+patch]
            raw = nan_to_eps(patch_slice)
            minmax = min_max_norm(raw)
            log1p_ = log1p(raw)
            clip_ = clip(log1p_)
            patch_steps = [raw, minmax, log1p_, clip_]
            for i, patch_img in enumerate(patch_steps):
                ax = axes[i, j]
                im = ax.imshow(patch_img, cmap=CMAP_)
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=14)
                if j == 0:
                    ax.set_ylabel(row_titles[i], fontsize=14)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(
            f"{data_plot_path}/{sample}_{subsample}_{resolution}_{chromosome}_{uuid}_patch{patch}.png", dpi=300)
        plt.close()


def prepare_triplates():
    for resolution in RESOLUTIONS:
        res_path = f"{OUTPUT_ROOT_PATH}/{resolution}"
        os.makedirs(res_path, exist_ok=True)
        for organism, samples in DATA.items():
            org_path = f"{res_path}/{organism}"
            os.makedirs(org_path, exist_ok=True)
            for sample, subsamples in samples.items():
                for subsample, content in subsamples.items():
                    for triplet in content["triplets"]:
                        print(
                            f"Processing {organism} > {sample} > {subsample} > {resolution} > {triplet}")

                        filepath0 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[0]}_{resolution}_KR.cool"
                        filepath1 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[1]}_{resolution}_KR.cool"
                        filepath2 = f"{ROOT_PATH}/{organism}/{sample}/{subsample}/{triplet[2]}_{resolution}_KR.cool"

                        cool_0 = cool.Cooler(filepath0)
                        cool_y = cool.Cooler(filepath1)
                        cool_1 = cool.Cooler(filepath2)
                        uuid = triplet[0] + "_" + triplet[1] + "_" + triplet[2]
                        import random
                        chroms = list(
                            zip(cool_y.chromnames, cool_y.chromsizes))
                        if not chroms:
                            continue
                        chromosome, chr_size = random.choice(chroms)
                        fetch = f"{chromosome}:{0}-{chr_size}"
                        mat_0 = cool_0.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        mat_y = cool_y.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        mat_1 = cool_1.matrix(
                            balance=BALANCE_COOL).fetch(fetch)
                        generate_patch(mat_0=mat_0, mat_y=mat_y, mat_1=mat_1,
                                       organism=organism, sample=sample, subsample=subsample, resolution=resolution, chromosome=chromosome, output_root_path=org_path, uuid=uuid)


if __name__ == "__main__":
    try:
        prepare_triplates()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
