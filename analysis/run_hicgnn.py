#!/usr/bin/env python3
"""Run HiCGNN on a genomic subregion of a Hi-C contact matrix (.npy / .txt)."""

import argparse
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from downstream_analysis.HiCGNN import hicgnn

EPOCHS = 50


def run_hicgnn(input, output, start, end):
    try:
        os.makedirs(output, exist_ok=True)
        _, ext = os.path.splitext(input)
        print("Extension:", ext)
        if ext.lower() == ".npy":
            full_matrix = np.load(input)
        elif ext.lower() == ".txt":
            full_matrix = np.loadtxt(input)
        else:
            raise ValueError(f"Unsupported matrix format '{ext}' (use .npy or .txt)")

        region = [start, end]
        matrix = full_matrix[region[0]:region[1], region[0]:region[1]]
        matrix_file = os.path.join(output, f"{region[0]}_{region[1]}.txt")
        np.savetxt(matrix_file, matrix, fmt="%.6f")

        print(f"Processing {matrix_file}")
        hicgnn.hicgnn(matrix_file, EPOCHS, output)
        print(f"Completed {matrix_file}")
    except Exception as ex:
        print(f"Exception\n: {ex}")
        traceback.print_exc()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix_file", required=True, help="Input .npy or .txt matrix")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--start", type=int, required=True, help="Region start bin (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Region end bin (exclusive)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_hicgnn(
        input=args.matrix_file,
        output=args.output_dir,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
