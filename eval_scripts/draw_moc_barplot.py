import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


NATURE_COLORS = ["#009e74",  "#0072b2",  "#f0e442", "#d55e00",
                 "#56b3e9", "#e69f00",  "#cc79a7", "#000000"
                 ]


# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv("/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/tads/moc.csv")

chromosomes = sorted(df["chromosome"].unique())
datasets = sorted(df["dataset"].unique())

# Colors for datasets
dataset_colors = {
    datasets[0]: NATURE_COLORS[0],
    datasets[1]: NATURE_COLORS[1]
}

# Colors for datasets
Title = {
    datasets[0]: "Different Timestamp (DMSO)",
    datasets[1]: "Different Dataset (HeLa-S3)"
}

box_width = 0.3

# -------------------------
# Loop over datasets
# -------------------------
for dataset in datasets:
    df_ds = df[df["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(3, 4))

    data_per_chr = []
    for chr_ in chromosomes:
        vals = df_ds[df_ds["chromosome"] == chr_]["moc"].values
        data_per_chr.append(vals)

    positions = np.arange(len(chromosomes))

    # Plot boxplot
    bp = ax.boxplot(
        data_per_chr,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False
    )

    # Color boxes
    for box in bp["boxes"]:
        box.set_facecolor(dataset_colors[dataset])
        box.set_alpha(0.7)

    # -------------------------
    # Annotate stats above each box
    # -------------------------
    # Calculate a dynamic spacing based on y-range
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    text_spacing = y_range * 0.05  # 5% of y-range between lines

    for i, vals in enumerate(data_per_chr):
        if len(vals) == 0:
            continue

        mean_val = np.mean(vals)
        median_val = np.median(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)

        # Start above the box
        y_start = max_val + text_spacing

        # ax.text(positions[i], y_start, f"max: {max_val:.2f}", ha='center', fontsize=9)
        # ax.text(positions[i], y_start + text_spacing, f"min: {min_val:.2f}", ha='center', fontsize=9)
        # ax.text(positions[i], y_start + 2*text_spacing, f"median: {median_val:.2f}", ha='center', fontsize=9)
        # ax.text(positions[i], y_start + 3*text_spacing, f"mean: {mean_val:.2f}", ha='center', fontsize=9)

    # -------------------------
    # Formatting
    # -------------------------
    ax.set_xticks(positions)
    ax.set_xticklabels(chromosomes, rotation=45)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("MoC")
    ax.set_title(f"{Title[dataset]}")

    # # Legend
    # ax.legend([plt.Line2D([0], [0], color=dataset_colors[dataset], lw=8)],
    #           [dataset], title="Dataset")

    fig.tight_layout()

    # -------------------------
    # Save figure
    # -------------------------
    out_name = f"/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/tads/moc_boxplot_{dataset}.png"
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"Saved: {out_name}")