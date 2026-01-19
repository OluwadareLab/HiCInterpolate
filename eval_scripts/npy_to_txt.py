import numpy as np
from pathlib import Path

BASE_INPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"
DATASETS = {
    # "dmso": "dmso",
    "hela_s3": "hela_s3"
}
GENOME_ID = "hg38"
CHROMOSOMES = [11, 13, 15, 17, 19, 21]
RESOLUTIONS = [10000]

if __name__ == "__main__":
    for dataset, rel_path in DATASETS.items():
        for chrom in CHROMOSOMES:
            sub_dir = f"{BASE_INPUT}/{rel_path}"
            for res in RESOLUTIONS:
                gt = f"{sub_dir}/chr{chrom}_y0.txt"
                y = np.loadtxt(gt)

                gt = f"{sub_dir}/chr{chrom}_y.txt"
                y = np.loadtxt(gt)

                gt = f"{sub_dir}/chr{chrom}_yt.txt"
                y = np.loadtxt(gt)

                gt = f"{sub_dir}/chr{chrom}_y.txt"
                y = np.loadtxt(gt)

