#!/usr/bin/env python3
"""Generate HiCCUPS loops from a Hi-C contact matrix (.npy / .txt) via juicer_tools."""

import argparse
import os
import subprocess
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JAR = os.path.join(_SCRIPT_DIR, "juicer_tools.2.20.00.jar")


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
    return mat


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def matrix_to_contact_list(
    matrix_file: str,
    contact_file: str,
    chrom: str,
    bin_size: int,
) -> int:
    mat = load_matrix(matrix_file)
    chrom_name = chrom_label(chrom)
    rows, cols = np.triu_indices_from(mat)
    values = mat[rows, cols]
    mask = values > 0
    rows, cols, values = rows[mask], cols[mask], values[mask]

    n_written = 0
    # juicer_tools pre short format + score:
    # <str1> <chr1> <pos1> <frag1> <str2> <chr2> <pos2> <frag2> <score>
    with open(contact_file, "w") as f:
        for i, j, v in zip(rows, cols, values):
            pos1 = int(i) * bin_size
            pos2 = int(j) * bin_size
            score = int(round(float(v)))
            if score <= 0:
                continue
            f.write(
                f"{chrom_name} {pos1} {chrom_name} {pos2} {score}\n"
            )
            n_written += 1
    return n_written


def run_cmd(cmd, step: str) -> None:
    print(f"Running {step}:\n  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_hiccups(
    matrix_file: str,
    output_dir: str,
    bin_size: int,
    chrom: str,
    genome_id: str,
    jar: str,
    threads: int,
    java_bin: str,
    norm: str,
    java_mem: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    chrom_name = chrom_label(chrom)
    contact_file = os.path.join(output_dir, f"{chrom_name}.txt")
    hic_file = os.path.join(output_dir, f"{chrom_name}.hic")
    loops_output_dir = os.path.join(output_dir, "loops")
    os.makedirs(loops_output_dir, exist_ok=True)

    if not os.path.isfile(jar):
        raise FileNotFoundError(f"juicer_tools jar not found: {jar}")
    if not os.path.isfile(matrix_file):
        raise FileNotFoundError(f"matrix file not found: {matrix_file}")

    print(f"Writing contact list -> {contact_file}", flush=True)
    n_contacts = matrix_to_contact_list(matrix_file, contact_file, chrom, bin_size)
    if n_contacts == 0:
        raise RuntimeError(f"No positive contacts written from {matrix_file}")
    print(f"Wrote {n_contacts} contacts", flush=True)

    java_prefix = [java_bin, f"-Xmx{java_mem}", "-jar", jar]

    # KR (default HiCCUPS norm) must exist in the .hic; NONE-only files fail
    # HiCCUPS postprocessing with "Data not available".
    cmd_pre = java_prefix + [
        "pre",
        contact_file,
        hic_file,
        genome_id,
        "-r", str(bin_size),
        "-c", chrom_name,
        "-k", norm,
    ]
    run_cmd(cmd_pre, "juicer_tools pre")
    print(f".hic generated: {hic_file}", flush=True)

    cmd_hiccups = java_prefix + [
        "hiccups",
        "--cpu",
        "--threads", str(threads),
        "-k", norm,
        "-c", chrom_name,
        "-r", str(bin_size),
        hic_file,
        loops_output_dir,
    ]
    run_cmd(cmd_hiccups, "juicer_tools hiccups")

    merged = os.path.join(loops_output_dir, "merged_loops.bedpe")
    if not os.path.isfile(merged) or os.path.getsize(merged) == 0:
        raise RuntimeError(
            f"HiCCUPS finished but no loops in {merged}. "
            f"Check norm={norm} exists in the .hic and sparsity."
        )
    print(f"HiCCUPS loops saved in {loops_output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate HiCCUPS loops from a Hi-C .npy/.txt contact matrix via juicer_tools."
    )
    parser.add_argument(
        "--matrix_file", required=True,
        help="Path to square contact matrix (.npy or .txt)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for contacts, .hic, and HiCCUPS loop output",
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
        "--genome_id", required=True,
        help="Genome id for juicer_tools pre (e.g. hg38, mm10)",
    )
    parser.add_argument(
        "--jar", default=_DEFAULT_JAR,
        help=f"Path to juicer_tools.jar (default: {_DEFAULT_JAR})",
    )
    parser.add_argument(
        "--threads", type=int, default=40,
        help="HiCCUPS CPU threads (default: 40)",
    )
    parser.add_argument(
        "--norm", default="KR", choices=["NONE", "VC", "VC_SQRT", "KR"],
        help="Normalization for pre + HiCCUPS (default: KR)",
    )
    parser.add_argument(
        "--java", dest="java_bin", default="java",
        help="Java executable (default: java)",
    )
    parser.add_argument(
        "--java_mem", default="16g",
        help="Java heap for juicer_tools (default: 16g)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_hiccups(
        matrix_file=args.matrix_file,
        output_dir=args.output_dir,
        bin_size=args.bin_size,
        chrom=args.chrom,
        genome_id=args.genome_id,
        jar=args.jar,
        threads=args.threads,
        java_bin=args.java_bin,
        norm=args.norm,
        java_mem=args.java_mem,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
