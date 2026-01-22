import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import *
import os
import numpy as np

COLORS = ["#009e74",  "#0072b2"]


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
    _, eigvecs = np.linalg.eigh(corr)
    pc1 = eigvecs[:, -1]
    return pc1


def compute_ab_compartment(hic_matrix, region):
    _, ext = os.path.splitext(hic_matrix)
    print("Extension:", ext)
    if ext.lower() == ".npy":
        hic = np.load(hic_matrix)
    if ext.lower() == ".txt":
        hic = np.loadtxt(hic_matrix)

    hic = hic[region[0]:region[1], region[0]:region[1]]

    oe = observed_expected(hic)
    corr, mask = correlation_matrix(oe)
    pc1_valid = compute_pc1(corr)

    pc1 = np.full(hic.shape[0], np.nan)
    pc1[mask] = pc1_valid

    return pc1


def plot_ab_track(pc1, resolution, filename):
    x = np.arange(len(pc1)) * resolution / 1e6

    plt.figure(figsize=(10, 2))
    plt.fill_between(x, pc1, 0, where=pc1 > 0,
                     color=COLORS[0], alpha=0.6, label="A")
    plt.fill_between(x, pc1, 0, where=pc1 < 0,
                     color=COLORS[1], alpha=0.6, label="B")

    plt.axhline(0, color="black", lw=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_ab_compartment(input: str, res: int, start: int, end: int, output: str):
    print(f"Processing: {input}...")
    output_file = os.path.join(output, f"ab_compartment.png")
    try:
        region = [start, end]
        pc1 = compute_ab_compartment(
            hic_matrix=input, region=region)
        plot_ab_track(pc1=pc1, resolution=res, filename=output_file)
    except ValueError as e:
        print(f"Error: {e}")
