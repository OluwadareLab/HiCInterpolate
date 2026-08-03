#!/usr/bin/env python3
"""A/B compartment analysis via cooltools (eigs_cis + saddle).

Pipeline follows:
  https://cooltools.readthedocs.io/en/latest/notebooks/compartments_and_saddles.html

  .npy -> .cool -> eigs_cis (E1) -> expected_cis -> saddle -> saddle_strength

Example:
  python analysis/run_compartment.py \\
    --input-dir /path/to/config_25k_64 \\
    --output-dir analysis/compartment_25k_64 \\
    --resolution 25000 --start-bp 20000000 --end-bp 60000000
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cooler
import cooltools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cooltools.api.saddle import saddle_strength
from matplotlib.colors import LogNorm

# Display order: Ground Truth, Ours, 4DMax, Linear, Optical Flow
METHODS = ("y", "pred", "4dmax", "linear", "of")
METHOD_LABELS = {
    "y": "Ground Truth",
    "pred": "Ours",
    "4dmax": "4DMax",
    "linear": "Linear",
    "of": "Optical Flow",
}
COLORS = {
    "y": "#000000",
    "pred": "#d55e00",
    "4dmax": "#cc79a7",
    "linear": "#0072b2",
    "of": "#009e74",
}
A_COLOR = "#d55e00"
B_COLOR = "#0072b2"
DPI = 300
Q_LO = 0.025
Q_HI = 0.975

_NAME_RE = re.compile(
    r"^(?P<res>\d+)_(?P<patch>\d+)_(?P<body>.+)_(?P<chrom>\d+)_(?P<method>y|pred|linear|of|4dmax)\.npy$"
)


def parse_filename(path: str) -> Optional[Dict[str, str]]:
    name = os.path.basename(path)
    m = _NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["path"] = path
    d["key"] = f"{d['body']}_{d['chrom']}"
    d["species"] = "human" if d["body"].startswith("human_") else (
        "mouse" if d["body"].startswith("mouse_") else "unknown"
    )
    return d


def discover_samples(input_dir: str) -> Dict[str, Dict[str, str]]:
    samples: Dict[str, Dict[str, str]] = defaultdict(dict)
    meta: Dict[str, Dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.npy"))):
        info = parse_filename(path)
        if info is None:
            continue
        samples[info["key"]][info["method"]] = path
        meta[info["key"]] = info
    return dict(samples), meta


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def write_unit_weights(cool_file: str) -> None:
    with h5py.File(cool_file, "r+") as h5:
        n = h5["bins/chrom"].shape[0]
        if "weight" in h5["bins"]:
            del h5["bins"]["weight"]
        h5["bins"].create_dataset("weight", data=np.ones(n, dtype=np.float64))


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


def load_full_matrix(path: str) -> np.ndarray:
    mat = np.load(path)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got {mat.shape} from {path}")
    mat = np.asarray(mat, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (mat + mat.T)


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


def gc_phasing_track(
    clr: cooler.Cooler,
    fasta_path: Optional[str],
    cache: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if not fasta_path or not os.path.isfile(fasta_path):
        return None
    key = f"{fasta_path}|{clr.binsize}|{','.join(clr.chromnames)}"
    if key in cache:
        return cache[key]
    import bioframe

    genome = bioframe.load_fasta(fasta_path)
    bins = clr.bins()[:][["chrom", "start", "end"]]
    # Restrict fasta chrom names present in cooler.
    chroms = set(clr.chromnames)
    genome = {k: v for k, v in genome.items() if k in chroms}
    if not genome:
        # try stripping/adding chr prefix
        alt = {}
        for k, v in bioframe.load_fasta(fasta_path).items():
            kk = k if k.startswith("chr") else f"chr{k}"
            if kk in chroms:
                alt[kk] = v
        genome = alt
    if not genome:
        return None
    gc_cov = bioframe.frac_gc(bins, genome)
    cache[key] = gc_cov
    return gc_cov


def flip_e1_to_reference(e1: np.ndarray, ref: np.ndarray) -> np.ndarray:
    valid = np.isfinite(e1) & np.isfinite(ref)
    if valid.sum() < 10:
        return e1
    if np.corrcoef(e1[valid], ref[valid])[0, 1] < 0:
        return -e1
    return e1


def analyze_cooler(
    clr: cooler.Cooler,
    view_df: pd.DataFrame,
    phasing_track: Optional[pd.DataFrame],
    n_groups: int,
    q_lo: float,
    q_hi: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return (eig_track, saddle_oe, interaction_sum, interaction_count)."""
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


def plot_e1_tracks(
    e1_by_method: Dict[str, pd.DataFrame],
    view_df: pd.DataFrame,
    out_path: str,
) -> None:
    methods = [m for m in METHODS if m in e1_by_method]
    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    start = int(view_df["start"].iloc[0])
    end = int(view_df["end"].iloc[0])
    for ax, method in zip(axes, methods):
        track = e1_by_method[method]
        mask = (
            (track["chrom"] == view_df["chrom"].iloc[0])
            & (track["start"] >= start)
            & (track["end"] <= end)
            & np.isfinite(track["E1"])
        )
        sub = track.loc[mask]
        x = (sub["start"].values + sub["end"].values) / 2 / 1e6
        y = sub["E1"].values
        ax.fill_between(x, y, 0, where=y > 0, color=A_COLOR, alpha=0.7, interpolate=True)
        ax.fill_between(x, y, 0, where=y < 0, color=B_COLOR, alpha=0.7, interpolate=True)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel(METHOD_LABELS[method], fontsize=9)
        ymax = float(np.nanmax(np.abs(y))) if y.size else 1.0
        ax.set_ylim(-ymax * 1.05, ymax * 1.05)
        ax.set_xlim(start / 1e6, end / 1e6)
    axes[-1].set_xlabel("Genomic position (Mb)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_saddles(
    saddles: Dict[str, np.ndarray],
    strengths: Dict[str, float],
    n_groups: int,
    q_lo: float,
    q_hi: float,
    out_path: str,
    vmin: float = 0.5,
    vmax: float = 2.0,
) -> None:
    """cooltools-style log OE saddles (trim outlier flanks)."""
    methods = [m for m in METHODS if m in saddles]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n + 1.2, 3.2), squeeze=False)
    edges = np.linspace(q_lo, q_hi, n_groups + 1)
    X, Y = np.meshgrid(edges, edges)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = None
    for i, (ax, method) in enumerate(zip(axes[0], methods)):
        C = np.asarray(saddles[method], dtype=float)
        if C.shape[0] == n_groups + 2:
            C = C[1:-1, 1:-1]
        im = ax.pcolormesh(X, Y, C, norm=norm, cmap="coolwarm", rasterized=True)
        ax.set_xlim(q_lo, q_hi)
        ax.set_ylim(q_hi, q_lo)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"{METHOD_LABELS[method]}\nS={strengths.get(method, float('nan')):.2f}"
        )
        ax.set_ylabel("B ← E1 → A" if i == 0 else "")
        ax.grid(False)
    fig.subplots_adjust(left=0.06, right=0.88, wspace=0.25, bottom=0.18, top=0.95)
    if im is not None:
        cax = fig.add_axes([0.90, 0.18, 0.015, 0.77])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("average OE")
        cbar.set_ticks([vmin, 1.0, vmax])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_strength_summary(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    by_method: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if np.isfinite(r["strength"]):
            by_method[r["method"]].append(r["strength"])
    methods = [m for m in METHODS if m in by_method]
    means = [float(np.mean(by_method[m])) for m in methods]
    stds = [float(np.std(by_method[m])) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, color=[COLORS[m] for m in methods], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Compartment strength (AA+BB)/(AB+BA)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    keys = sorted({r["key"] for r in rows})
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(keys) * len(methods)), 4.5))
    width = 0.8 / max(len(methods), 1)
    lookup = {(r["key"], r["method"]): r["strength"] for r in rows}
    for i, method in enumerate(methods):
        vals = [lookup.get((k, method), np.nan) for k in keys]
        ax.bar(
            np.arange(len(keys)) + i * width,
            vals,
            width=width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.85,
        )
    ax.set_xticks(np.arange(len(keys)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(keys, rotation=90, fontsize=7)
    ax.set_ylabel("Compartment strength")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path.replace(".png", "_per_sample.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_summary(rows: List[Dict], out_path: str) -> None:
    by_method: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["method"] == "y":
            continue
        if np.isfinite(r.get("e1_pearson", np.nan)):
            by_method[r["method"]].append(r["e1_pearson"])
    methods = [m for m in METHODS if m != "y" and m in by_method]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    means = [float(np.mean(by_method[m])) for m in methods]
    stds = [float(np.std(by_method[m])) for m in methods]
    ax.bar(x, means, yerr=stds, color=[COLORS[m] for m in methods], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("E1 Pearson vs Ground Truth")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def e1_agreement(track: pd.DataFrame, ref: pd.DataFrame) -> Dict[str, float]:
    a = track["E1"].to_numpy()
    b = ref["E1"].to_numpy()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return {"pearson": float("nan"), "sign_agree": float("nan")}
    return {
        "pearson": float(np.corrcoef(a[valid], b[valid])[0, 1]),
        "sign_agree": float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))),
    }


def process_sample(
    key: str,
    method_paths: Dict[str, str],
    meta: Dict[str, str],
    out_dir: str,
    resolution: int,
    start_bp: int,
    end_bp: int,
    methods: Tuple[str, ...],
    n_groups: int,
    q_lo: float,
    q_hi: float,
    strength_extent: int,
    balance: bool,
    fasta_human: Optional[str],
    fasta_mouse: Optional[str],
    gc_cache: Dict[str, pd.DataFrame],
    keep_cool: bool,
) -> List[Dict]:
    sample_dir = os.path.join(out_dir, key)
    os.makedirs(sample_dir, exist_ok=True)
    cool_dir = os.path.join(sample_dir, "cool") if keep_cool else tempfile.mkdtemp(prefix="cool_")
    os.makedirs(cool_dir, exist_ok=True)

    chrom = meta["chrom"]
    species = meta.get("species", "unknown")
    fasta = fasta_human if species == "human" else fasta_mouse if species == "mouse" else None

    e1_tracks: Dict[str, pd.DataFrame] = {}
    saddles: Dict[str, np.ndarray] = {}
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}
    strengths: Dict[str, float] = {}
    rows: List[Dict] = []
    ref_e1 = None
    view_df = None

    try:
        for method in methods:
            if method not in method_paths:
                continue
            mat = load_full_matrix(method_paths[method])
            cool_file = os.path.join(cool_dir, f"{method}.cool")
            clr = matrix_to_cool(mat, cool_file, chrom, resolution, balance=balance)
            if view_df is None:
                view_df = make_view_df(clr, chrom, start_bp, end_bp)
            phasing = gc_phasing_track(clr, fasta, gc_cache)

            eig_track, saddle_oe, S, C = analyze_cooler(
                clr, view_df, phasing, n_groups, q_lo, q_hi
            )

            if ref_e1 is not None:
                e1 = eig_track["E1"].to_numpy().copy()
                flipped = flip_e1_to_reference(e1, ref_e1["E1"].to_numpy())
                if np.nanmean(flipped * e1) < 0:
                    S = np.asarray(S)[::-1, ::-1].copy()
                    C = np.asarray(C)[::-1, ::-1].copy()
                    saddle_oe = np.asarray(saddle_oe)[::-1, ::-1].copy()
                eig_track = eig_track.copy()
                eig_track["E1"] = flipped
            elif method == "y":
                ref_e1 = eig_track

            strength = strength_at_extent(S, C, strength_extent)
            e1_tracks[method] = eig_track
            saddles[method] = saddle_oe
            sums[method] = S
            counts[method] = C
            strengths[method] = strength

            eig_track.to_csv(os.path.join(sample_dir, f"{method}_E1.tsv"), sep="\t", index=False)
            np.save(os.path.join(sample_dir, f"{method}_saddle.npy"), saddle_oe)
            np.save(os.path.join(sample_dir, f"{method}_saddle_sum.npy"), S)
            np.save(os.path.join(sample_dir, f"{method}_saddle_count.npy"), C)

            if ref_e1 is not None and method != "y":
                agree = e1_agreement(eig_track, ref_e1)
            else:
                agree = {"pearson": 1.0 if method == "y" else float("nan"), "sign_agree": 1.0 if method == "y" else float("nan")}

            rows.append(
                {
                    "key": key,
                    "method": method,
                    "strength": strength,
                    "e1_pearson": agree["pearson"],
                    "e1_sign_agree": agree["sign_agree"],
                    "n_e1": int(np.isfinite(eig_track["E1"]).sum()),
                }
            )
            print(
                f"  [{key}/{method}] strength={strength:.3f} e1_r={agree['pearson']:.3f}",
                flush=True,
            )

        if e1_tracks and view_df is not None:
            plot_e1_tracks(e1_tracks, view_df, os.path.join(sample_dir, "e1_tracks.png"))
        if saddles:
            plot_saddles(
                saddles,
                strengths,
                n_groups,
                q_lo,
                q_hi,
                os.path.join(sample_dir, "saddle_plots.png"),
            )
    finally:
        if not keep_cool and os.path.isdir(cool_dir):
            shutil.rmtree(cool_dir, ignore_errors=True)

    return rows


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fields = ["key", "method", "strength", "e1_pearson", "e1_sign_agree", "n_e1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    default_in = (
        "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
        "datasets/timeseries/full_triplets/output/config_25k_64"
    )
    default_hg38 = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/hg38.fa"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=default_in)
    p.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "compartment_25k_64"),
    )
    p.add_argument("--resolution", type=int, default=25000)
    p.add_argument("--start-bp", type=int, default=20_000_000)
    p.add_argument("--end-bp", type=int, default=60_000_000)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--keys", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=38, help="Digitized E1 groups (cooltools default-ish)")
    p.add_argument("--q-lo", type=float, default=Q_LO)
    p.add_argument("--q-hi", type=float, default=Q_HI)
    p.add_argument(
        "--strength-extent",
        type=int,
        default=4,
        help="Extent index into cooltools saddle_strength profile",
    )
    p.add_argument("--balance", action="store_true", help="Run ICE balancing (default: unit weights)")
    p.add_argument("--keep-cool", action="store_true", help="Keep per-sample .cool files")
    p.add_argument("--fasta-human", default=default_hg38 if os.path.isfile(default_hg38) else None)
    p.add_argument("--fasta-mouse", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    samples, meta = discover_samples(args.input_dir)
    if not samples:
        raise SystemExit(f"No matching .npy samples in {args.input_dir}")

    keys = sorted(samples)
    if args.keys:
        want = set(args.keys)
        keys = [k for k in keys if k in want]
    if args.max_samples is not None:
        keys = keys[: args.max_samples]

    print(
        f"cooltools {cooltools.__version__} | region {args.start_bp}-{args.end_bp} | "
        f"{len(keys)}/{len(samples)} samples",
        flush=True,
    )
    if args.fasta_human:
        print(f"Human GC fasta: {args.fasta_human}", flush=True)
    if args.fasta_mouse:
        print(f"Mouse GC fasta: {args.fasta_mouse}", flush=True)

    gc_cache: Dict[str, pd.DataFrame] = {}
    all_rows: List[Dict] = []
    for key in keys:
        print(f"Processing {key}", flush=True)
        try:
            rows = process_sample(
                key=key,
                method_paths=samples[key],
                meta=meta[key],
                out_dir=args.output_dir,
                resolution=args.resolution,
                start_bp=args.start_bp,
                end_bp=args.end_bp,
                methods=tuple(args.methods),
                n_groups=args.n_groups,
                q_lo=args.q_lo,
                q_hi=args.q_hi,
                strength_extent=args.strength_extent,
                balance=args.balance,
                fasta_human=args.fasta_human,
                fasta_mouse=args.fasta_mouse,
                gc_cache=gc_cache,
                keep_cool=args.keep_cool,
            )
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  FAILED {key}: {exc}", flush=True)

    csv_path = os.path.join(args.output_dir, "compartment_metrics.csv")
    write_csv(all_rows, csv_path)
    plot_strength_summary(all_rows, os.path.join(args.output_dir, "strength_summary.png"))
    plot_agreement_summary(all_rows, os.path.join(args.output_dir, "e1_agreement_summary.png"))
    print(f"Wrote metrics -> {csv_path}", flush=True)
    print(f"Plots -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
