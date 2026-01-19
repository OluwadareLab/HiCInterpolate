import os
import numpy as np
import subprocess

BASE_INPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data"
DATASETS = {
    "dmso": "dmso",
    "hela_s3": "hela_s3"
}
RESOLUTIONS = [10000]
# CHROMOSOMES = [11, 13, 15, 17, 19, 21]
# FILE_SUFFIX = ["yt", "y"]
# CHROMOSOMES = [11]
# FILE_SUFFIX = ["y0", "y1"]

CHROMOSOMES = [13, 15, 17, 19, 21]
FILE_SUFFIX = ["y", "yt"]

GENOME_ID = "hg38"
JAVA_JAR = "juicer_tools.2.20.00.jar"
BASE_OUTPUT = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/hiccups_loops"


def matrix_to_contact_list(matrix_file, contact_file, chrom, bin_size, max_val=-1.0):
    mat = np.loadtxt(matrix_file)
    if (max_val > 0):
        mat = mat * max_val

    rows, cols = np.triu_indices_from(mat)
    values = mat[rows, cols]

    with open(contact_file, "w") as f:
        for i, j, v in zip(rows, cols, values):
            if v != 0:
                pos1 = i * bin_size
                pos2 = j * bin_size
                f.write(f"chr{chrom} {pos1} chr{chrom} {pos2} {v:.6f}\n")


def run_hiccups(matrix_file, output_dir, bin_size, chrom, data_type, java_jar, max_val=-1.0):
    contact_file = os.path.join(output_dir, f"chr{chrom}_{data_type}.contacts")
    matrix_to_contact_list(matrix_file=matrix_file, contact_file=contact_file,
                           chrom=chrom, bin_size=bin_size, max_val=max_val)

    hic_file = os.path.join(output_dir, f"chr{chrom}_{data_type}.hic")
    cmd_pre = [
        "java", "-jar", java_jar,
        "pre",
        contact_file,
        hic_file,
        GENOME_ID,
        "-r", str(bin_size),
        "-c", f"chr{chrom}"
    ]
    print("Running juicer_tools pre...")
    subprocess.run(cmd_pre, check=True)
    print(f".hic generated: {hic_file}")
    loops_output_dir = os.path.join(output_dir, "loops")
    os.makedirs(loops_output_dir, exist_ok=True)
    cmd_hiccups = [
        "java", "-jar", java_jar,
        "hiccups",
        "--cpu",
        "--threads", "40",
        "-c", f"chr{chrom}",
        "-r", str(bin_size),
        hic_file,
        loops_output_dir
    ]
    print("Running juicer_tools hiccups...")
    subprocess.run(cmd_hiccups, check=True)
    print(f"HICCUPS loops saved in {output_dir}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_OUTPUT), exist_ok=True)
    for dataset, rel_path in DATASETS.items():
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

                    run_hiccups(matrix_file=file_path, output_dir=out_dir,
                                bin_size=res, chrom=chrom, data_type=suffix, java_jar=JAVA_JAR)

                print("All chromosomes processed successfully!")
