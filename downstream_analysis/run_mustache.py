import os
import shutil
import subprocess

import numpy as np
import pandas as pd

try:
    import cooler
except ImportError as exc:
    raise SystemExit("cooler is required: pip install cooler") from exc


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


def _write_unit_weights(cool_file: str) -> None:
    import h5py

    with h5py.File(cool_file, "r+") as h5:
        n = h5["bins/chrom"].shape[0]
        if "weight" in h5["bins"]:
            del h5["bins"]["weight"]
        h5["bins"].create_dataset("weight", data=np.ones(n, dtype=np.float64))


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

    if balance:
        try:
            cooler.balance_cooler(cool_file, store=True, ignore_diags=2)
        except Exception:
            _write_unit_weights(cool_file)
    else:
        _write_unit_weights(cool_file)

    return cool_file


def run_mustache(
    matrix_file: str,
    output_dir: str,
    bin_size: int,
    chrom: str,
    p_threshold: float = 0.1,
    sparsity_threshold: float = 0.88,
    processes: int = 4,
    balance: bool = False,
    mustache_bin: str = None,
    keep_cool: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    chrom_name = chrom_label(chrom)
    cool_file = os.path.join(output_dir, f"{chrom_name}.cool")
    loops_file = os.path.join(output_dir, f"{chrom_name}.mustache.tsv")

    if not os.path.isfile(matrix_file):
        raise FileNotFoundError(f"matrix file not found: {matrix_file}")

    mustache_exe = mustache_bin or shutil.which("mustache")
    if mustache_exe is None:
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
    subprocess.run(cmd, check=True)

    if not os.path.isfile(loops_file):
        raise RuntimeError(f"Mustache finished but output missing: {loops_file}")

    if not keep_cool and os.path.isfile(cool_file):
        os.remove(cool_file)
