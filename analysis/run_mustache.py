#!/usr/bin/env python3
"""Predict chromatin loops with Mustache from a Hi-C contact matrix (.npy / .txt).

Pipeline: matrix -> .cool (cooler) -> Mustache loop calls (.tsv).
Requires: numpy, pandas, cooler, mustache-hic (CLI: ``mustache``).
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    import cooler
except ImportError as exc:
    raise SystemExit(
        "cooler is required: pip install cooler"
    ) from exc


def load_matrix(matrix_file: str) -> np.ndarray:
    _, ext = os.path.splitext(matrix_file)
    ext = ext.lower()
    if ext == ".npy":
        mat = np.load(matrix_file)
    elif ext == ".txt":
        mat = np.loadtxt(matrix_file)
    else:
        raise ValueError(f"Unsupported matrix format '{ext}' (use .npy or .txt)")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {mat.shape}")
    return mat.astype(np.float64, copy=False)


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def resolution_label(bin_size: int) -> str:
    if bin_size % 1000 == 0:
        return f"{bin_size // 1000}kb"
    return str(bin_size)


def matrix_to_cool(
    matrix_file: str,
    cool_file: str,
    chrom: str,
    bin_size: int,
    balance: bool,
) -> str:
    mat = load_matrix(matrix_file)
    n = mat.shape[0]
    chrom_name = chrom_label(chrom)

    bins = pd.DataFrame(
        {
            "chrom": [chrom_name] * n,
            "start": np.arange(n, dtype=np.int64) * bin_size,
            "end": np.arange(1, n + 1, dtype=np.int64) * bin_size,
        }
    )

    rows, cols = np.triu_indices(n)
    values = mat[rows, cols]
    mask = values > 0
    if not np.any(mask):
        raise RuntimeError(f"No positive contacts in {matrix_file}")

    pixels = pd.DataFrame(
        {
            "bin1_id": rows[mask].astype(np.int64),
            "bin2_id": cols[mask].astype(np.int64),
            "count": values[mask].astype(np.float64),
        }
    )

    if os.path.exists(cool_file):
        os.remove(cool_file)

    cooler.create_cooler(
        cool_file,
        bins=bins,
        pixels=pixels,
        dtypes={"count": np.float64},
        assembly=None,
        ordered=True,
    )
    print(f".cool written: {cool_file} ({len(pixels)} pixels, {n} bins)", flush=True)

    if balance:
        try:
            cooler.balance_cooler(cool_file, store=True, ignore_diags=2)
            print("ICE balance stored as bins/weight", flush=True)
        except Exception as exc:
            print(f"WARNING: balance failed ({exc}); using weight=1", flush=True)
            _write_unit_weights(cool_file)
    else:
        _write_unit_weights(cool_file)

    return cool_file


def _write_unit_weights(cool_file: str) -> None:
    import h5py

    with h5py.File(cool_file, "r+") as h5:
        n = h5["bins/chrom"].shape[0]
        if "weight" in h5["bins"]:
            del h5["bins"]["weight"]
        h5["bins"].create_dataset("weight", data=np.ones(n, dtype=np.float64))
    print("Wrote unit weights (bins/weight=1)", flush=True)


def run_cmd(cmd, step: str) -> None:
    print(f"Running {step}:\n  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_mustache(
    matrix_file: str,
    output_dir: str,
    bin_size: int,
    chrom: str,
    p_threshold: float,
    sparsity_threshold: float,
    processes: int,
    balance: bool,
    mustache_bin: str,
    keep_cool: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    chrom_name = chrom_label(chrom)
    cool_file = os.path.join(output_dir, f"{chrom_name}.cool")
    loops_file = os.path.join(output_dir, f"{chrom_name}.mustache.tsv")

    if not os.path.isfile(matrix_file):
        raise FileNotFoundError(f"matrix file not found: {matrix_file}")

    mustache_exe = mustache_bin or shutil.which("mustache")
    if mustache_exe is None:
        # pip install mustache-hic exposes `mustache`
        raise FileNotFoundError(
            "mustache CLI not found on PATH; install with: pip install mustache-hic"
        )

    matrix_to_cool(matrix_file, cool_file, chrom, bin_size, balance=balance)

    res = resolution_label(bin_size)
    cmd = [
        mustache_exe,
        "-f", cool_file,
        "-ch", chrom_name,
        "-r", res,
        "-pt", str(p_threshold),
        "-st", str(sparsity_threshold),
        "-p", str(processes),
        "-o", loops_file,
    ]
    run_cmd(cmd, "mustache")

    if not os.path.isfile(loops_file):
        raise RuntimeError(f"Mustache finished but output missing: {loops_file}")
    print(f"Mustache loops saved: {loops_file}", flush=True)

    if not keep_cool and os.path.isfile(cool_file):
        os.remove(cool_file)
        print(f"Removed intermediate .cool: {cool_file}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hi-C .npy/.txt matrix to .cool and call loops with Mustache."
        )
    )
    parser.add_argument(
        "--matrix_file", required=True,
        help="Path to square contact matrix (.npy or .txt)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for .cool (optional) and Mustache .tsv output",
    )
    parser.add_argument(
        "--bin_size", type=int, required=True,
        help="Matrix resolution in bp (e.g. 10000)",
    )
    parser.add_argument(
        "--chrom", required=True,
        help="Chromosome id/name (e.g. 11 or chr11)",
    )
    parser.add_argument(
        "--p_threshold", type=float, default=0.1,
        help="Mustache p-value / FDR threshold -pt (default: 0.1)",
    )
    parser.add_argument(
        "--sparsity_threshold", type=float, default=0.88,
        help="Mustache sparsity threshold -st (default: 0.88; try 0.7 if sparse)",
    )
    parser.add_argument(
        "--processes", type=int, default=4,
        help="Mustache parallel processes -p (default: 4)",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Run cooler ICE balance and store weights (default: unit weights)",
    )
    parser.add_argument(
        "--keep_cool", action="store_true",
        help="Keep intermediate .cool file in output_dir",
    )
    parser.add_argument(
        "--mustache", dest="mustache_bin", default=None,
        help="Path to mustache executable (default: mustache on PATH)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_mustache(
        matrix_file=args.matrix_file,
        output_dir=args.output_dir,
        bin_size=args.bin_size,
        chrom=args.chrom,
        p_threshold=args.p_threshold,
        sparsity_threshold=args.sparsity_threshold,
        processes=args.processes,
        balance=args.balance,
        mustache_bin=args.mustache_bin,
        keep_cool=args.keep_cool,
    )


if __name__ == "__main__":
    main()
