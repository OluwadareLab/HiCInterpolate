import os
import cooler
import numpy as np
import subprocess
import argparse
import pandas as pd

JAVA_JAR = "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/HiCInterpolate/analysis/juicer_tools.2.20.00.jar"


def matrix_to_contact_list(matrix_file, inter_filename_prefix, chrom, bin_size):
    _, ext = os.path.splitext(matrix_file)
    print("Extension:", ext)
    if ext.lower() == ".npy":
        mat = np.load(matrix_file)
    if ext.lower() == ".txt":
        mat = np.loadtxt(matrix_file)

    rows, cols = np.triu_indices_from(mat)
    values = mat[rows, cols]

    with open(f"{inter_filename_prefix}.txt", "w") as f:
        for i, j, v in zip(rows, cols, values):
            if v != 0:
                pos1 = i * bin_size
                pos2 = j * bin_size
                f.write(f"chr{chrom} {pos1} chr{chrom} {pos2} {v:.6f}\n")


def matrix_to_cool(matrix_file, inter_filename_prefix, chrom, bin_size):
    _, ext = os.path.splitext(matrix_file)
    print("Extension:", ext)
    if ext.lower() == ".npy":
        mat = np.load(matrix_file)
    if ext.lower() == ".txt":
        mat = np.loadtxt(matrix_file)

    n_bins = mat.shape[0]

    bins = pd.DataFrame({
        'chrom': [chrom] * n_bins,
        'start': np.arange(0, n_bins * bin_size, bin_size),
        'end': np.arange(bin_size, (n_bins + 1) * bin_size, bin_size)
    })

    i, j = np.triu_indices(n_bins)
    counts = mat[i, j]

    mask = counts > 0
    pixels = pd.DataFrame({
        'bin1_id': i[mask],
        'bin2_id': j[mask],
        'count': counts[mask]
    })

    cooler.create_cooler(f'{inter_filename_prefix}.cool',
                         bins=bins, pixels=pixels)


def run_hiccups(matrix_file, output_dir, bin_size, chrom, genome_id):
    inter_filename_prefix = os.path.join(output_dir, f"chr{chrom}")
    matrix_to_cool(matrix_file=matrix_file, inter_filename_prefix=inter_filename_prefix,
                   chrom=chrom, bin_size=bin_size)

    # hic_file = os.path.join(output_dir, f"chr{chrom}.hic")
    # cmd_pre = [
    #     "java", "-jar", JAVA_JAR,
    #     "pre",
    #     f"{inter_filename_prefix}.txt",
    #     hic_file,
    #     genome_id,
    #     "-r", str(bin_size),
    #     "-c", f"chr{chrom}",
    #     "-k", "NONE"
    # ]
    # print("Running juicer_tools pre...")
    # subprocess.run(cmd_pre, check=True)
    # print(f".hic generated: {hic_file}")
    # loops_output_dir = os.path.join(output_dir, "loops")
    # os.makedirs(loops_output_dir, exist_ok=True)
    # cmd_hiccups = [
    #     "java", "-jar", JAVA_JAR,
    #     "hiccups",
    #     "--cpu",
    #     "--threads", "40",
    #     "-c", f"chr{chrom}",
    #     "-r", str(bin_size),
    #     hic_file,
    #     loops_output_dir
    # ]
    # print("Running juicer_tools hiccups...")
    # subprocess.run(cmd_hiccups, check=True)
    # print("HiCCUPS complete")
    # print(f"HICCUPS loops saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HiCCUPS loops from a contact matrix")
    parser.add_argument("--matrix_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bin_size", type=int, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--genome_id", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[run_hiccups.py] start: {args.matrix_file}")
    run_hiccups(
        matrix_file=args.matrix_file,
        output_dir=args.output_dir,
        bin_size=args.bin_size,
        chrom=args.chrom,
        genome_id=args.genome_id,
    )
    print(f"[run_hiccups.py] done: {args.output_dir}")


if __name__ == "__main__":
    main()
