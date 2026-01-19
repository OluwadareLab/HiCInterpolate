import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from hmmlearn import hmm
from scipy.sparse import csr_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import *
import os
import numpy as np


BASE_INPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"
DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
RESOLUTIONS = [10000]
REGIONS = [
    [6000, 6500],
    [8900, 9400]
]
# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["yt", "y"]
CHROMOSOMES = [11]
FILE_SUFFIX = ["y0", "yt", "y", "y1"]

# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["y0", "y", "yt", "y1"]

NATURE_COLORS = ["#009e74",  "#0072b2",  "#f0e442", "#d55e00",
                 "#56b3e9", "#e69f00",  "#cc79a7", "#000000"
                 ]

BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/ab_compartments"


def valid_bins(matrix, eps=1e-6):
    std = np.nanstd(matrix, axis=1)
    return std > eps


def observed_expected(matrix, eps=1e-6):
    n = matrix.shape[0]
    oe = np.zeros_like(matrix, dtype=float)

    for d in range(n):
        diag = np.diag(matrix, k=d)
        if len(diag) == 0:
            continue
        mean = np.nanmean(diag)
        if mean < eps:
            continue

        val = diag / mean
        oe += np.diag(val, k=d)
        if d > 0:
            oe += np.diag(val, k=-d)

    return oe


def correlation_matrix(oe, eps=1e-6):
    mask = valid_bins(oe, eps)
    oe_valid = oe[mask][:, mask]

    np.fill_diagonal(oe_valid, 0.0)

    corr = np.corrcoef(oe_valid)
    corr = np.nan_to_num(corr)

    return corr, mask


def compute_pc1(corr):
    # symmetric matrix → eigh is stable
    eigvals, eigvecs = np.linalg.eigh(corr)

    # largest eigenvalue → last eigenvector
    pc1 = eigvecs[:, -1]
    return pc1


def compute_ab_compartment(hic_matrix, region):
    hic = np.loadtxt(hic_matrix)
    hic = hic[region[0]:region[1], region[0]:region[1]]

    oe = observed_expected(hic)
    corr, mask = correlation_matrix(oe)
    pc1_valid = compute_pc1(corr)

    # reinsert into full-length vector
    pc1 = np.full(hic.shape[0], np.nan)
    pc1[mask] = pc1_valid

    return pc1


def plot_ab_track(pc1, resolution, chrom, filename):
    x = np.arange(len(pc1)) * resolution / 1e6

    plt.figure(figsize=(10, 2))
    plt.fill_between(x, pc1, 0, where=pc1 > 0,
                     color=NATURE_COLORS[1], alpha=0.6, label="A")
    plt.fill_between(x, pc1, 0, where=pc1 < 0,
                     color=NATURE_COLORS[3], alpha=0.6, label="B")

    plt.axhline(0, color="black", lw=0.5)
    # plt.xlabel(f"chr{chrom} position (Mb)")
    # plt.ylabel("PC1")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_OUTPUT), exist_ok=True)
    for (dataset, rel_path), region in zip(DATASETS.items(), REGIONS):
        for chrom in CHROMOSOMES:
            for res in RESOLUTIONS:
                in_sub_dir = os.path.join(BASE_INPUT, rel_path)
                out_sub_dir = os.path.join(BASE_OUTPUT, rel_path)
                os.makedirs(out_sub_dir, exist_ok=True)
                for suffix in FILE_SUFFIX:
                    file_path = f"{in_sub_dir}/chr{chrom}_{suffix}.txt"
                    print(f"Processing: {file_path}...")

                    out_dir = os.path.join(
                        out_sub_dir, f"chr{chrom}_{suffix}")
                    os.makedirs(out_dir, exist_ok=True)
                    output_file = os.path.join(
                        out_dir, f"chr{chrom}_{suffix}.png")

                    try:
                        pc1 = compute_ab_compartment(
                            hic_matrix=file_path, region=region)
                        plot_ab_track(pc1=pc1, resolution=res,
                                      chrom=chrom, filename=output_file)
                    except ValueError as e:
                        print(
                            f"Skipping chr{chrom}_{suffix} due to error: {e}")

                print("All chromosomes processed successfully!")
