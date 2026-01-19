import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


BASE_INPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"
DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
RESOLUTIONS = [10000]

# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["y", "yt"]
# CHROMOSOMES = [11, 21]
# FILE_SUFFIX = ["y0", "y1"]

CHROMOSOMES = [11]
FILE_SUFFIX = ["y0", "y", "yt", "y1"]
REGIONS = [
    [6000, 6500],
    [8900, 9400]
]

# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["y0", "y", "yt", "y1"]


BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/hic_maps"

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['figure.dpi'] = 300
# CMAP_ = mcolors.LinearSegmentedColormap.from_list(
#     'juicebox', ['#FFFFFF', '#FF0000']
# )

CMAP_JUICEBOX = mcolors.LinearSegmentedColormap.from_list(
    "juicebox",
    [
        "#fee8c8",
        "#fdbb84",
        "#e34a33",
        "#b30000"
    ],
    N=256
)


def draw_single_hic_map(matrix: np.ndarray, filename: str):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

    mat = matrix.copy()
    mat[mat <= 0] = np.nan

    # Juicebox-like dynamic range
    vmin = np.nanpercentile(mat, 5)
    vmax = np.nanpercentile(mat, 99)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        mat,
        cmap=CMAP_JUICEBOX,
        norm=norm,
        interpolation="nearest"
    )

    ax.axis("off")

    # # Juicebox-style slim colorbar
    # cbar = fig.colorbar(
    #     im,
    #     ax=ax,
    #     fraction=0.03,
    #     pad=0.02
    # )
    # cbar.ax.tick_params(labelsize=6, length=2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6, length=2)

    plt.tight_layout(pad=0.1)
    plt.savefig(f"{filename}.png", dpi=300)
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
                    matrix = np.loadtxt(file_path)
                    draw_single_hic_map(
                        matrix[region[0]:region[1], region[0]:region[1]], filename=f"{out_sub_dir}/chr{chrom}_{suffix}")
