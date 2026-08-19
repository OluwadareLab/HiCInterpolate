import os
import tempfile
import warnings
from typing import Optional, Tuple

import cooler
import cooltools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cooltools.api.saddle import saddle_strength

COLORS = ["#009e74", "#0072b2"]
Q_LO = 0.025
Q_HI = 0.975
N_GROUPS = 38
STRENGTH_EXTENT = 4


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def write_unit_weights(cool_file: str) -> None:
    with h5py.File(cool_file, "r+") as h5:
        n = h5["bins/chrom"].shape[0]
        if "weight" in h5["bins"]:
            del h5["bins"]["weight"]
        h5["bins"].create_dataset("weight", data=np.ones(n, dtype=np.float64))


def load_matrix(path: str) -> np.ndarray:
    _, ext = os.path.splitext(path)
    print("Extension:", ext)
    ext = ext.lower()
    if ext == ".npy":
        mat = np.load(path)
    elif ext == ".txt":
        mat = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported matrix format '{ext}' (use .npy or .txt)")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got {mat.shape} from {path}")
    mat = np.asarray(mat, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (mat + mat.T)


def matrix_to_cool(
    matrix: np.ndarray,
    cool_file: str,
    chrom: str,
    bin_size: int,
    balance: bool = False,
) -> cooler.Cooler:
    n = matrix.shape[0]
    chrom_name = chrom_label(chrom)
    bins = pd.DataFrame(
        {
            "chrom": [chrom_name] * n,
            "start": np.arange(n, dtype=np.int64) * bin_size,
            "end": np.arange(1, n + 1, dtype=np.int64) * bin_size,
        }
    )
    rows, cols = np.triu_indices(n)
    values = matrix[rows, cols]
    mask = values > 0
    if not np.any(mask):
        raise RuntimeError("No positive contacts")
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
        ordered=True,
    )
    if balance:
        try:
            cooler.balance_cooler(cool_file, store=True, ignore_diags=2)
        except Exception:
            write_unit_weights(cool_file)
    else:
        write_unit_weights(cool_file)
    return cooler.Cooler(cool_file)


def make_view_df(clr: cooler.Cooler, chrom: str, start_bp: int, end_bp: int) -> pd.DataFrame:
    chrom_name = chrom_label(chrom)
    chrom_end = int(clr.chromsizes[chrom_name])
    s = max(0, int(start_bp))
    e = min(int(end_bp), chrom_end)
    if e - s < 50 * clr.binsize:
        raise ValueError(f"Region too small on {chrom_name}: {s}-{e} (chrom end {chrom_end})")
    return pd.DataFrame(
        {"chrom": [chrom_name], "start": [s], "end": [e], "name": [f"{chrom_name}:{s}-{e}"]}
    )


def analyze_cooler(
    clr: cooler.Cooler,
    view_df: pd.DataFrame,
    phasing_track: Optional[pd.DataFrame],
    n_groups: int,
    q_lo: float,
    q_hi: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cis_eigs = cooltools.eigs_cis(
            clr,
            phasing_track,
            view_df=view_df,
            n_eigs=1,
        )
    eig_track = cis_eigs[1][["chrom", "start", "end", "E1"]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvd = cooltools.expected_cis(clr=clr, view_df=view_df)
        interaction_sum, interaction_count = cooltools.saddle(
            clr,
            cvd,
            eig_track,
            "cis",
            n_bins=n_groups,
            qrange=(q_lo, q_hi),
            view_df=view_df,
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        saddle_oe = interaction_sum / interaction_count
    return eig_track, saddle_oe, interaction_sum, interaction_count


def strength_at_extent(S: np.ndarray, C: np.ndarray, extent: int) -> float:
    profile = saddle_strength(S, C)
    if profile is None or len(profile) == 0:
        return float("nan")
    idx = min(max(int(extent), 0), len(profile) - 1)
    val = profile[idx]
    return float(val) if np.isfinite(val) else float("nan")


def compute_ab_compartment(hic_matrix, region, bin_size, chrom="chr1", balance=False):
    hic = load_matrix(hic_matrix)
    if region is not None and region[0] is not None and region[1] is not None:
        hic = hic[region[0]:region[1], region[0]:region[1]]

    cool_file = os.path.join(tempfile.mkdtemp(prefix="ab_cool_"), "matrix.cool")
    try:
        clr = matrix_to_cool(hic, cool_file, chrom, bin_size, balance=balance)
        chrom_end = int(clr.chromsizes[chrom_label(chrom)])
        view_df = make_view_df(clr, chrom, 0, chrom_end)
        eig_track, _, S, C = analyze_cooler(
            clr, view_df, None, N_GROUPS, Q_LO, Q_HI
        )
        _ = strength_at_extent(S, C, STRENGTH_EXTENT)
        return eig_track["E1"].to_numpy()
    finally:
        if os.path.isfile(cool_file):
            os.remove(cool_file)
        cool_dir = os.path.dirname(cool_file)
        if os.path.isdir(cool_dir):
            try:
                os.rmdir(cool_dir)
            except OSError:
                pass


def plot_ab_track(pc1, resolution, filename):
    x = np.arange(len(pc1)) * resolution / 1e6

    plt.figure(figsize=(10, 2))
    plt.fill_between(x, pc1, 0, where=pc1 > 0,
                     color=COLORS[0], alpha=0.6, label="A")
    plt.fill_between(x, pc1, 0, where=pc1 < 0,
                     color=COLORS[1], alpha=0.6, label="B")

    plt.axhline(0, color="black", lw=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_ab_compartment(input: str, res: int, start: int, end: int, output: str, chrom: str = None):
    print(f"Processing: {input}...")
    os.makedirs(output, exist_ok=True)
    output_file = os.path.join(output, "ab_compartment.png")
    try:
        region = [start, end]
        pc1 = compute_ab_compartment(
            hic_matrix=input,
            region=region,
            bin_size=res,
            chrom=chrom or "chr1",
        )
        plot_ab_track(pc1=pc1, resolution=res, filename=output_file)
    except ValueError as e:
        print(f"Error: {e}")
