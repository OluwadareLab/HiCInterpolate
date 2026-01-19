import os
import csv
import pandas as pd
import tad_scores


BASE_INPUT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data"
RESOLUTION = 10_000
DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
CHROMOSOMES = [11, 13, 15, 17, 19, 21]
MATRIX_PREFIXES = ['y', 'yt']
TOOLS_SUBFOLDER = {
    'EmbedTAD': 'embedtad',
    'TopDom': 'topdom',
    'Spectral': 'spectral_tads'
}
TOOLS_FILE_EXT = {
    'EmbedTAD': 'txt',
    'TopDom': 'txt.bed',
    'Spectral': 'bed'
}


BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/tads"
os.makedirs(BASE_OUTPUT, exist_ok=True)

nature_colors = ["#009e74",  "#0072b2",  "#f0e442", "#d55e00",
                 "#56b3e9", "#e69f00",  "#cc79a7", "#000000"
                 ]

if __name__ == "__main__":
    fieldnames = ['dataset', 'chromosome', 'tool', 'moc']
    csv_file = f"{BASE_OUTPUT}/moc.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames)
            writer.writeheader()

    for dataset, rel_path in DATASETS.items():
        for chrom in CHROMOSOMES:
            for (tool, tool_subfolder), (tool_name, tool_ext) in zip(TOOLS_SUBFOLDER.items(), TOOLS_FILE_EXT.items()):
                sub_dir = os.path.join(
                    BASE_INPUT_PATH, tool_subfolder, rel_path)
                if tool == "EmbedTAD":
                    y_file = os.path.join(
                        sub_dir, f"chr{chrom}_y.{tool_ext}")
                    yt_file = os.path.join(
                        sub_dir, f"chr{chrom}_yt.{tool_ext}")
                elif tool == "Spectral":
                    y_file = os.path.join(
                        sub_dir, f"chr{chrom}_y", f"Spectral.{tool_ext}")
                    yt_file = os.path.join(
                        sub_dir, f"chr{chrom}_yt", f"Spectral.{tool_ext}")
                else:
                    y_file = os.path.join(
                        sub_dir, f"chr{chrom}_y", f"chr{chrom}_y.{tool_ext}")
                    yt_file = os.path.join(
                        sub_dir, f"chr{chrom}_yt", f"chr{chrom}_yt.{tool_ext}")
                moc = 0
                try:

                    if tool == "Spectral":
                        y_tads = pd.read_csv(y_file, sep=",", header=None)
                        yt_tads = pd.read_csv(yt_file, sep=",", header=None)
                    else:
                        y_tads = pd.read_csv(y_file, sep="\t", header=None)
                        yt_tads = pd.read_csv(yt_file, sep="\t", header=None)

                    if tool == "TopDom":
                        y_tads = (y_tads.loc[y_tads.iloc[:, 3]
                                             == "domain", y_tads.columns[[1, 2]]].reset_index(drop=True).set_axis([0, 1], axis=1))
                        yt_tads = (yt_tads.loc[yt_tads.iloc[:, 3]
                                               == "domain", yt_tads.columns[[1, 2]]].reset_index(drop=True).set_axis([0, 1], axis=1))

                    moc = tad_scores.get_moc(tads=y_tads, true_tads=yt_tads)

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

                except Exception as e:
                    print(
                        f"Error processing {dataset} chr{chrom} with tool {tool}: {e}")
