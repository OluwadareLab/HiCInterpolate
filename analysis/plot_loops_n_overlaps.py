import os
import csv
import pandas as pd
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import plot_loops as loops_vis


FITHIC_BASE_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/fithic/results"
HICCUPS_BASE_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/hiccups/output"
DATASETS = {
    "dmso_control": "dmso_control",
    # "dtag_v1": "dtag_v1",
    # "hct116_noatp30": "hct116_noatp30m",
    # "hct116_notranscription60m": "hct116_notranscription60m",
    "hela_s3": "hela_s3",
}
CHROMOSOMES = [11, 21]
CHROMOSOME_SIZE = [135086622, 46709983]
REGIONS = [[6000, 7000], [3000, 4000]]
BIN_SIZE = 10_000
GENOME_ID = "hg38"
MATRIX_PREFIXES = ['y', 'hici']

BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/HiCInterpolate/analysis/loops"
os.makedirs(BASE_OUTPUT, exist_ok=True)

nature_colors = ["#009e74",  "#0072b2",  "#f0e442", "#d55e00",
                 "#56b3e9", "#e69f00",  "#cc79a7", "#000000"
                 ]


def find_overlaps(dataset1, dataset2):
    overlaps = []
    for _, row1 in dataset1.iterrows():
        overlap = dataset2[(dataset2["fragmentMid1"] == row1["fragmentMid1"])
                           & (dataset2["fragmentMid2"] == row1["fragmentMid2"])]
        if not overlap.empty:
            overlaps.append(f"{row1['fragmentMid1']}-{row1['fragmentMid2']}")
    return set(overlaps)


def find_overlaps_loops(y_file, hici_file):
    cols = ['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2']

    df1 = pd.read_csv(y_file, sep='\t', names=cols)
    df2 = pd.read_csv(hici_file, sep='\t', names=cols)
    overlap = df1.merge(
        df2,
        on=['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2'],
        how='inner'
    )

    return len(overlap)


def find_overlaps(dataset1, dataset2):
    overlaps = []
    for _, row1 in dataset1.iterrows():
        overlap = dataset2[(dataset2["fragmentMid1"] == row1["fragmentMid1"])
                           & (dataset2["fragmentMid2"] == row1["fragmentMid2"])]
        if not overlap.empty:
            overlaps.append(f"{row1['fragmentMid1']}-{row1['fragmentMid2']}")
    return set(overlaps)


def draw_overlaps(y_dataset, hici_dataset, output_file):
    overlaps = find_overlaps(y_dataset, hici_dataset)
    print(f"Ground Truth intersection Predicted: {len(overlaps)}")

    only_gt = len(y_dataset) - len(overlaps)
    print(f"Only GT: {only_gt}")

    only_hici = len(hici_dataset) - len(overlaps)
    print(f"Only Predicted: {only_hici}")

    fig, ax = plt.subplots(figsize=(6, 8))

    radius = 0.3

    # Vertical positions (top, bottom)
    positions = [(0.5, 0.65), (0.5, 0.35)]

    # Draw circles
    for pos, color in zip(positions, nature_colors):
        circle = Circle(pos, radius, alpha=0.5, color=color)
        ax.add_patch(circle)

    # Labels
    ax.text(0.5, 1.0, "Ground Truth", fontsize=14,
            ha="center", va="center", color=nature_colors[7])

    ax.text(0.5, 0.0, "HiCInterpolate", fontsize=14,
            ha="center", va="center", color=nature_colors[7])

    # Counts
    ax.text(0.5, 0.78, str(only_gt), fontsize=13,
            ha="center", va="center", color=nature_colors[7])
    ax.text(0.5, 0.22, str(only_hici), fontsize=13,
            ha="center", va="center", color=nature_colors[7])

    ax.text(0.5, 0.50, str(len(overlaps)), fontsize=15,
            ha="center", va="center", color=nature_colors[7], fontweight="bold")

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")

    # plt.title("Overlap", fontsize=16)
    plt.savefig(f"{output_file}",
                dpi=300, bbox_inches="tight")

    return len(overlaps)


if __name__ == "__main__":
    fieldnames = ['dataset', 'type', 'chromosome',
                  'tool', 'count', 'percent', 'recovery_rate']
    csv_file = f"{BASE_OUTPUT}/recovery_rate.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames)
            writer.writeheader()

    for dataset, rel_path in DATASETS.items():
        for chrom, chrom_size, region in zip(CHROMOSOMES, CHROMOSOME_SIZE, REGIONS):
            sub_dir = f"{rel_path}_chr{chrom}"
            y_contacts_file = os.path.join(
                FITHIC_BASE_PATH, sub_dir, "y", "FitHiC.spline_pass1.res10000.significances.txt.gz")
            print(f"Reading: {y_contacts_file}")
            y_contacts = pd.read_csv(y_contacts_file, sep="\t")
            y_contacts = y_contacts[y_contacts['p-value'] < 0.05]
            hici_contacts_file = os.path.join(
                FITHIC_BASE_PATH, sub_dir, "hici", "FitHiC.spline_pass1.res10000.significances.txt.gz")
            print(f"Reading: {hici_contacts_file}")
            hici_contacts = pd.read_csv(hici_contacts_file, sep="\t")
            hici_contacts = hici_contacts[hici_contacts['p-value'] < 0.05]

            output_path = f"{BASE_OUTPUT}/{sub_dir}"
            os.makedirs(output_path, exist_ok=True)

            # overlap_contacts_plot_filename = f"{output_path}/chr{chrom}_contacts_overlaps.png"

            # print(f"Plotting contacts recovery rate...")
            # overlap_count = draw_overlaps(y_contacts, hici_contacts,
            #                               overlap_contacts_plot_filename)

            # print(f"Calculating contacts recovery rate...")
            # contacts_recovery_rate = (
            #     overlap_count / len(y_contacts)) * 100
            # print(f"Contacts recovery rate: {contacts_recovery_rate}%")

            y_loop_file = os.path.join(
                HICCUPS_BASE_PATH, sub_dir, "y", "merged_loops.bedpe")
            # print(f"Reading: {y_loop_file}")
            # y_loops = pd.read_csv(y_loop_file, sep="\t")
            # y_loops = y_loops[y_loops['fdrDonut'] < 0.001]

            hici_loop_file = os.path.join(
                HICCUPS_BASE_PATH, sub_dir, "hici", "merged_loops.bedpe")
            # print(f"Reading: {hici_loop_file}")
            # hici_loops = pd.read_csv(hici_loop_file, sep="\t")
            # hici_loops = hici_loops[hici_loops['fdrDonut'] < 0.001]

            # print(f"Calculating loops recovery rate...")
            # overlap_loop_count = find_overlaps_loops(
            #     y_loop_file, hici_loop_file)
            # loops_recovery_rate = (overlap_loop_count / len(y_loops)) * 100
            # print(f"Loops recovery rate: {loops_recovery_rate}%")

            # print(f"Writing recovery rates to CSV: {csv_file}")
            # with open(csv_file, 'a', newline='') as f:
            #     writer = csv.DictWriter(
            #         f, fieldnames=fieldnames)
            #     writer.writerow({
            #         'dataset': dataset,
            #         'type': 'y',
            #         'chromosome': chrom,
            #         'tool': 'FitHiC',
            #         'count': len(y_contacts),
            #         'percent': -1,
            #         'recovery_rate': -1
            #     })

            #     writer.writerow({
            #         'dataset': dataset,
            #         'type': 'hici',
            #         'chromosome': chrom,
            #         'tool': 'FitHiC',
            #         'count': len(hici_contacts),
            #         'percent': (len(hici_contacts)/len(y_contacts))*100,
            #         'recovery_rate': contacts_recovery_rate
            #     })

            #     writer.writerow({
            #         'dataset': dataset,
            #         'type': 'y',
            #         'chromosome': chrom,
            #         'tool': 'HiCCUPS',
            #         'count': len(y_loops),
            #         'percent': -1,
            #         'recovery_rate': -1
            #     })

            #     writer.writerow({
            #         'dataset': dataset,
            #         'type': 'hici',
            #         'chromosome': chrom,
            #         'tool': 'HiCCUPS',
            #         'count': len(hici_loops),
            #         'percent': (len(hici_loops)/len(y_loops))*100,
            #         'recovery_rate': loops_recovery_rate
            #     })

            #     f.close()

            print(f"Plotting loops...")
            y_loops_plot_filename = f"{output_path}/y_loops.png"
            y_matrix = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/raw_predicted_data/{sub_dir}/y_chr{chrom}.txt"
            vis = loops_vis.Triangle(y_matrix, BIN_SIZE,
                                     f"chr{chrom}", region[0]*BIN_SIZE, region[1]*BIN_SIZE)
            vis.matrix_plot()
            vis.plot_loops(y_loop_file, marker_color=nature_colors[0])
            print(f'Writing -> {y_loops_plot_filename}')
            vis.outfig(y_loops_plot_filename)

            hici_loops_plot_filename = f"{output_path}/hici_loops.png"
            hici_matrix = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/raw_predicted_data/{sub_dir}/hici_chr{chrom}.txt"
            vis = loops_vis.Triangle(hici_matrix, BIN_SIZE,
                                     f"chr{chrom}", region[0]*BIN_SIZE, region[1]*BIN_SIZE)
            vis.matrix_plot()
            vis.plot_loops(hici_loop_file, marker_color=nature_colors[0])
            print(f'Writing -> {hici_loops_plot_filename}')
            vis.outfig(hici_loops_plot_filename)
