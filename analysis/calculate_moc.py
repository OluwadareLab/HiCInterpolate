import os
import csv
import pandas as pd
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import plot_loops as loops_vis
import tad_scores


BASE_INPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/raw_predicted_data/tads"
DATASETS = {
    "dmso_control": "dmso_control",
    # "dtag_v1": "dtag_v1",
    # "hct116_noatp30": "hct116_noatp30m",
    # "hct116_notranscription60m": "hct116_notranscription60m",
    "hela_s3": "hela_s3",
}
CHROMOSOMES = [11, 21]
CHROMOSOME_SIZE = [135086622, 46709983]
REGIONS = [
    [5000, 5200],
    # [6050, 6150],
    [1000, 4000]
]
BIN_SIZE = 10_000
GENOME_ID = "hg38"
MATRIX_PREFIXES = ['y', 'hici']

BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/analysis/tads"
os.makedirs(BASE_OUTPUT, exist_ok=True)

nature_colors = ["#009e74",  "#0072b2",  "#f0e442", "#d55e00",
                 "#56b3e9", "#e69f00",  "#cc79a7", "#000000"
                 ]

TOOLS = ['EmbedTAD', 'TopDom', 'Armatus', 'Spectral']

if __name__ == "__main__":
    fieldnames = ['dataset', 'chromosome', 'tool', 'moc']
    csv_file = f"{BASE_OUTPUT}/moc.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames)
            writer.writeheader()

    for dataset, rel_path in DATASETS.items():
        for chrom, chrom_size, region in zip(CHROMOSOMES, CHROMOSOME_SIZE, REGIONS):
            sub_dir = f"{rel_path}_chr{chrom}"
            print(f"Processing {sub_dir}...")
            for tool in TOOLS:
                y_tads_file = os.path.join(
                    BASE_INPUT_PATH, sub_dir, "y", f"{tool}.bed")
                y_tads = pd.read_csv(y_tads_file, sep=",", header=None)
                hici_tads_file = os.path.join(
                    BASE_INPUT_PATH, sub_dir, "hici", f"{tool}.bed")
                hici_tads = pd.read_csv(hici_tads_file, sep=",", header=None)
                moc = tad_scores.get_moc(tads=y_tads, true_tads=hici_tads)
                print(f"Tool: {tool}, MoC: {moc}")
                print(f"Writing recovery rates to CSV: {csv_file}")
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(
                        f, fieldnames=fieldnames)
                    writer.writerow({
                        'dataset': dataset,
                        'chromosome': chrom,
                        'tool': tool,
                        'moc': moc
                    })
