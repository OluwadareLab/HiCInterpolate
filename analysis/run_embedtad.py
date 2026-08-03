#!/usr/bin/env python3
"""Call EmbedTAD TAD detection on a Hi-C contact matrix (.npy / .txt)."""

import argparse
import os
import subprocess
import sys

import numpy as np


def run_embedtad(matrix_file, output_dir, bin_size):
    os.makedirs(output_dir, exist_ok=True)
    _, ext = os.path.splitext(matrix_file)
    print("Extension:", ext)
    mat_file = matrix_file
    if ext.lower() == ".npy":
        data = np.load(matrix_file, allow_pickle=True)
        mat_file = f"{output_dir}/matrix.txt"
        np.savetxt(mat_file, data)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    embedtad = os.path.join(repo_root, "downstream_analysis", "EmbedTAD", "embedtad.py")

    cmd = [
        sys.executable,
        embedtad,
        "--input", f"{mat_file}",
        "--output", f"{output_dir}",
        "--resolution", f"{bin_size}",
        "--worker", "CPU",
        "--normalization", "True",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix_file", required=True, help="Input .npy or .txt matrix")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--bin_size", type=int, required=True, help="Bin size / resolution")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_embedtad(
        matrix_file=args.matrix_file,
        output_dir=args.output_dir,
        bin_size=args.bin_size,
    )


if __name__ == "__main__":
    main()
