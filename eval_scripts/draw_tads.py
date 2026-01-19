import os
import csv
from numpy import rint
import pandas as pd
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import loops_tad_draw_lib as loops_vis


MATRIX_BASE_INPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"
LOOPS_BASE_INPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/tads"

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
GENOME_ID = "hg38"

LOOPS_BASE_OUTPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/tads"


NATURE_COLORS = ["#009e74",  "#0072b2",  "#f0e442", "#d55e00",
                 "#56b3e9", "#e69f00",  "#cc79a7", "#000000"
                 ]


if __name__ == "__main__":
    os.makedirs(os.path.join(LOOPS_BASE_OUTPUT_PATH), exist_ok=True)
    fieldnames = ['dataset', 'type', 'chromosome',
                  'tool', 'count', 'percent', 'recovery_rate']
    csv_file = f"{LOOPS_BASE_OUTPUT_PATH}/recovery_rate.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames)
            writer.writeheader()

    for (dataset, rel_path), region in zip(DATASETS.items(), REGIONS):
        for chrom in CHROMOSOMES:
            for res in RESOLUTIONS:
                in_sub_dir = os.path.join(LOOPS_BASE_INPUT_PATH, rel_path)
                out_sub_dir = os.path.join(LOOPS_BASE_OUTPUT_PATH, rel_path)
                os.makedirs(out_sub_dir, exist_ok=True)
                for suffix in FILE_SUFFIX:
                    loop_file = os.path.join(
                        out_sub_dir, f"chr{chrom}_{suffix}.txt")
                    print(f"Processing: {loop_file}...")

                    out_dir = os.path.join(
                        out_sub_dir, f"chr{chrom}_{suffix}")
                    os.makedirs(out_dir, exist_ok=True)

                    print(f"Plotting loops...")
                    loop_fig_filename = os.path.join(
                        out_dir, f"chr{chrom}_{suffix}.png")

                    matrix_filename = os.path.join(
                        MATRIX_BASE_INPUT_PATH, rel_path, f"chr{chrom}_{suffix}.txt")

                    vis = loops_vis.Triangle(matrix_filename, res,
                                             f"chr{chrom}", region[0]*res, region[1]*res)
                    vis.matrix_plot()
                    vis.plot_TAD(loop_file, resol=res, line_color=NATURE_COLORS[1])
                    print(f'Writing -> {loop_fig_filename}')
                    vis.outfig(loop_fig_filename)
