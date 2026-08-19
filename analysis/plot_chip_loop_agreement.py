#!/usr/bin/env python3
"""CTCF ChIP-seq agreement of Mustache chromatin loops.

A loop is CTCF-valid when both anchors overlap at least one CTCF peak
from the dataset peak BED.

Also reports recovery of ground-truth CTCF-validated loops: fraction of
GT CTCF-valid loops that exactly match a loop called by each method.

Supports human (DMSO/dTAG) and mouse cerebellar granule neuron.

Writes per-chromosome and summary CSVs, plus Nature-style bar plots
(counts and GT-valid recovery %, 300 dpi PNG, no title).

Example:
  /opt/miniconda3/envs/hicexplorer/bin/python analysis/plot_chip_loop_agreement.py
  /opt/miniconda3/envs/hicexplorer/bin/python analysis/plot_chip_loop_agreement.py --dataset cerebellar
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("y", "pred", "linear", "of", "4dmax")
METHOD_LABELS = {
    "y": "Ground Truth",
    "pred": "HiCInterpolate",
    "linear": "HL",
    "of": "HOF",
    "4dmax": "4DMax",
}
# Okabe–Ito / Nature colorblind palette
METHOD_COLORS = {
    "y": "#000000",
    "pred": "#009E73",
    "4dmax": "#CC79A7",
    "linear": "#0072B2",
    "of": "#E69F00",
}

DEFAULT_PEAK_BED_HUMAN = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/chip_seq/hg38_htert_rep1_ctcf.bed"
)
DEFAULT_PEAK_BED_CEREBELLAR = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/chip_seq/"
    "mm10_cerebella_granule_cell_ctcf.bed"
)

# Per-dataset: samples, organism, chrom splits, default peak BED, output subdir
DATASETS = {
    "human": {
        "samples": {
            "DMSO": {"mustache_key": "dmso_control"},
            "dTAG": {"mustache_key": "dtag_v1"},
        },
        "organism": "human",
        "chrom_splits": {
            "Seen": (11, 16, 21),
            "Unseen": (10, 15, 20),
        },
        "peak_bed": DEFAULT_PEAK_BED_HUMAN,
        "output_subdir": "chip_loop_agreement",
    },
    "cerebellar": {
        "samples": {
            "Cerebellar": {
                "mustache_key": "cerebellar_granule_neuron_control",
            },
        },
        "organism": "mouse",
        "chrom_splits": {
            "Cross-organism": (10, 15, 19),
        },
        "peak_bed": DEFAULT_PEAK_BED_CEREBELLAR,
        "output_subdir": "chip_loop_agreement_cerebellar",
    },
}

DIR_RE = re.compile(
    r"^(?P<res>\d+)_(?P<organism>human|mouse)_(?P<body>.+)_(?P<chrom>\d+)_"
    r"(?P<method>y|pred|linear|of|4dmax)$"
)

DPI = 300

Loop = Tuple[str, int, int, str, int, int]
PeaksByChrom = Dict[str, List[Tuple[int, int]]]


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
        }
    )


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def load_peaks(bed_path: str) -> PeaksByChrom:
    """Load BED peaks; merge overlaps per chromosome; sort by start."""
    peaks: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = chrom_label(parts[0])
            start, end = int(parts[1]), int(parts[2])
            if end <= start:
                continue
            peaks[chrom].append((start, end))

    merged: PeaksByChrom = {}
    for chrom, intervals in peaks.items():
        intervals.sort()
        out: List[Tuple[int, int]] = []
        for s, e in intervals:
            if not out or s > out[-1][1]:
                out.append((s, e))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], e))
        merged[chrom] = out
    return merged


def anchor_overlaps_peak(
    peaks: PeaksByChrom,
    chrom: str,
    start: int,
    end: int,
) -> bool:
    """True if [start, end) overlaps any merged peak on chrom."""
    ivals = peaks.get(chrom)
    if not ivals or end <= start:
        return False
    i = bisect.bisect_left(ivals, (end, -1)) - 1
    return i >= 0 and ivals[i][1] > start


def load_loops(tsv_path: str) -> List[Loop]:
    if not os.path.isfile(tsv_path) or os.path.getsize(tsv_path) == 0:
        return []
    df = pd.read_csv(tsv_path, sep="\t")
    required = [
        "BIN1_CHR",
        "BIN1_START",
        "BIN1_END",
        "BIN2_CHROMOSOME",
        "BIN2_START",
        "BIN2_END",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{tsv_path}: missing columns {missing}")
    loops = []
    for row in df.itertuples(index=False):
        loops.append(
            (
                chrom_label(getattr(row, "BIN1_CHR")),
                int(getattr(row, "BIN1_START")),
                int(getattr(row, "BIN1_END")),
                chrom_label(getattr(row, "BIN2_CHROMOSOME")),
                int(getattr(row, "BIN2_START")),
                int(getattr(row, "BIN2_END")),
            )
        )
    return loops


def find_mustache_tsv(run_dir: str) -> Optional[str]:
    if not os.path.isdir(run_dir):
        return None
    cands = [
        os.path.join(run_dir, n)
        for n in os.listdir(run_dir)
        if n.endswith(".mustache.tsv")
    ]
    if not cands:
        return None
    cands.sort()
    return cands[0]


def discover_runs(
    mustache_root: str,
    samples: Dict[str, Dict[str, str]],
    organism: str,
) -> Dict[Tuple[str, int, str], str]:
    """(sample_key, chrom, method) -> tsv_path."""
    out: Dict[Tuple[str, int, str], str] = {}
    for name in sorted(os.listdir(mustache_root)):
        path = os.path.join(mustache_root, name)
        if not os.path.isdir(path):
            continue
        m = DIR_RE.match(name)
        if not m or m.group("organism") != organism:
            continue
        body = m.group("body")
        sample_key = None
        for label, meta in samples.items():
            key = meta["mustache_key"]
            if key in body:
                sample_key = label
                break
        if sample_key is None:
            continue
        tsv = find_mustache_tsv(path)
        if tsv is None:
            continue
        chrom = int(m.group("chrom"))
        method = m.group("method")
        out[(sample_key, chrom, method)] = tsv
    return out


def score_loops(
    loops: Sequence[Loop],
    peaks: PeaksByChrom,
) -> List[Dict]:
    rows: List[Dict] = []
    for c1, s1, e1, c2, s2, e2 in loops:
        a1 = anchor_overlaps_peak(peaks, c1, s1, e1)
        a2 = anchor_overlaps_peak(peaks, c2, s2, e2)
        if a1 and a2:
            status = "both"
        elif a1 or a2:
            status = "single"
        else:
            status = "none"
        rows.append(
            {
                "chr1": c1,
                "start1": s1,
                "end1": e1,
                "chr2": c2,
                "start2": s2,
                "end2": e2,
                "anchor1_pass": int(a1),
                "anchor2_pass": int(a2),
                "status": status,
                "valid_chip": int(status == "both"),
            }
        )
    return rows


def summarize(detail: List[Dict]) -> Dict[str, float]:
    n = len(detail)
    n_both = sum(1 for r in detail if r["status"] == "both")
    n_single = sum(1 for r in detail if r["status"] == "single")
    n_none = sum(1 for r in detail if r["status"] == "none")
    pct = (100.0 * n_both / n) if n else 0.0
    return {
        "n_loops": n,
        "n_both_anchors": n_both,
        "n_single_anchor": n_single,
        "n_no_anchor": n_none,
        "pct_valid_chip": pct,
    }


def gt_valid_set(detail: List[Dict]) -> Set[Loop]:
    return {
        (r["chr1"], r["start1"], r["end1"], r["chr2"], r["start2"], r["end2"])
        for r in detail
        if r["status"] == "both"
    }


def recovery_vs_gt_valid(
    method_loops: Sequence[Loop],
    gt_valid: Set[Loop],
) -> Tuple[int, int, float]:
    """Return (n_overlap, n_gt_valid, pct_recovered)."""
    n_gt = len(gt_valid)
    if n_gt == 0:
        return 0, 0, 0.0
    n_ov = len(set(method_loops) & gt_valid)
    return n_ov, n_gt, 100.0 * n_ov / n_gt


def _plot_chrom_method_bars(
    ax,
    per_chrom_rows: List[Dict],
    sample: str,
    split: str,
    chroms: Sequence[int],
    methods: Sequence[str],
    value_key: str,
    ylabel: str,
    show_ylabel: bool,
    value_fmt: str = "int",
    ylim: Optional[Tuple[float, float]] = None,
) -> List:
    """Draw grouped bars; return legend handles from the last method pass."""
    lookup = {
        (r["chromosome"], r["method"]): r[value_key]
        for r in per_chrom_rows
        if r["sample"] == sample and r["split"] == split
    }
    x = np.arange(len(chroms), dtype=float)
    n_m = len(methods)
    width = min(0.16, 0.8 / max(n_m, 1))
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    ymax = 0.0
    handles = []

    for i, method in enumerate(methods):
        vals = [float(lookup.get((chrom_label(c), method), 0)) for c in chroms]
        ymax = max(ymax, max(vals) if vals else 0.0)
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.95,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            label=METHOD_LABELS[method],
            alpha=0.92,
        )
        handles.append(bars)
        for bar, val in zip(bars, vals):
            if val <= 0:
                continue
            if value_fmt == "pct":
                label = f"{val:.0f}"
            else:
                label = f"{int(round(val))}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="black",
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([chrom_label(c) for c in chroms])
    ax.set_xlabel(f"{sample} ({split})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(1.0, ymax) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    return handles


def _write_sample_figures(
    per_chrom_rows: List[Dict],
    output_dir: str,
    samples: Sequence[str],
    chrom_splits: Dict[str, Sequence[int]],
    value_key: str,
    ylabel: str,
    file_stem: str,
    value_fmt: str = "int",
    fixed_ylim: Optional[Tuple[float, float]] = None,
) -> List[str]:
    apply_nature_style()
    methods = list(METHODS)
    written: List[str] = []
    os.makedirs(output_dir, exist_ok=True)

    for sample in samples:
        for split, chroms in chrom_splits.items():
            if not chroms:
                continue
            fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
            handles = _plot_chrom_method_bars(
                ax,
                per_chrom_rows,
                sample=sample,
                split=split,
                chroms=chroms,
                methods=methods,
                value_key=value_key,
                ylabel=ylabel,
                show_ylabel=True,
                value_fmt=value_fmt,
                ylim=fixed_ylim,
            )
            fig.legend(
                handles,
                [METHOD_LABELS[m] for m in methods],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.14),
                ncol=5,
                frameon=False,
                columnspacing=1.0,
                handlelength=1.2,
                handletextpad=0.4,
            )
            split_slug = split.lower().replace(" ", "_").replace("-", "_")
            panel = os.path.join(
                output_dir,
                f"{file_stem}_{sample.lower()}_{split_slug}.png",
            )
            fig.savefig(panel, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            written.append(panel)

    return written


def plot_comparison(
    per_chrom_rows: List[Dict],
    output_dir: str,
    samples: Sequence[str],
    chrom_splits: Dict[str, Sequence[int]],
) -> List[str]:
    """Per-split count plots + GT ChIP-seq validated recovery % plots."""
    written = []
    written.extend(
        _write_sample_figures(
            per_chrom_rows,
            output_dir,
            samples=samples,
            chrom_splits=chrom_splits,
            value_key="n_both_anchors",
            ylabel="CTCF ChIP-valid loops",
            file_stem="chip_loop_agreement",
            value_fmt="int",
        )
    )
    written.extend(
        _write_sample_figures(
            per_chrom_rows,
            output_dir,
            samples=samples,
            chrom_splits=chrom_splits,
            value_key="pct_gt_valid_recovered",
            ylabel="GT ChIP-Valid loops recovery (%)",
            file_stem="chip_loop_gt_valid_recovery",
            value_fmt="pct",
            fixed_ylim=(0, 108),
        )
    )
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "CTCF ChIP agreement of Mustache loops via peak BED "
            "(both anchors must overlap a peak), plus recovery of "
            "ground-truth CTCF-validated loops. "
            "Datasets: human (DMSO/dTAG), cerebellar (mouse neuron)."
        )
    )
    p.add_argument(
        "--dataset",
        choices=("human", "cerebellar", "all"),
        default="all",
        help="Which dataset(s) to analyze (default: all)",
    )
    p.add_argument(
        "--mustache_dir",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/mustache",
    )
    p.add_argument(
        "--peak_bed",
        default=None,
        help="Override CTCF peak BED for the selected dataset(s)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override output dir (default: <analysis>/<dataset output_subdir>)",
    )
    p.add_argument(
        "--write_loop_csv",
        action="store_true",
        help="Write per-loop validity CSVs (one per sample/chrom/method)",
    )
    return p.parse_args(argv)


def run_dataset(
    dataset_name: str,
    mustache_dir: str,
    peak_bed: Optional[str],
    output_dir: Optional[str],
    write_loop_csv: bool,
    script_dir: str,
) -> None:
    cfg = DATASETS[dataset_name]
    samples = cfg["samples"]
    chrom_splits: Dict[str, Sequence[int]] = {
        k: tuple(v) for k, v in cfg["chrom_splits"].items()
    }
    bed = peak_bed or cfg["peak_bed"]
    out_dir = output_dir or os.path.join(script_dir, cfg["output_subdir"])
    os.makedirs(out_dir, exist_ok=True)
    detail_dir = os.path.join(out_dir, "per_loop")
    if write_loop_csv:
        os.makedirs(detail_dir, exist_ok=True)

    if not os.path.isfile(bed):
        raise SystemExit(f"Missing peak BED: {bed}")

    peaks = load_peaks(bed)
    n_peaks = sum(len(v) for v in peaks.values())
    print(
        f"[{dataset_name}] Loaded {n_peaks} merged peaks from {bed}",
        flush=True,
    )

    runs = discover_runs(mustache_dir, samples, cfg["organism"])
    if not runs:
        raise SystemExit(
            f"[{dataset_name}] No Mustache runs found under {mustache_dir}"
        )

    chrom_split: Dict[int, str] = {}
    for split_name, chroms in chrom_splits.items():
        for c in chroms:
            chrom_split[c] = split_name
    target_chroms = sorted(chrom_split)

    per_chrom_rows: List[Dict] = []
    agg_counts: Dict[Tuple[str, str, str], List[Tuple[int, int, int, int]]] = (
        defaultdict(list)
    )

    for sample in samples:
        for chrom in target_chroms:
            split = chrom_split[chrom]
            scored: Dict[str, Tuple[List[Loop], List[Dict], Dict[str, float]]] = {}
            for method in METHODS:
                key = (sample, chrom, method)
                tsv = runs.get(key)
                if tsv is None:
                    print(
                        f"SKIP missing {sample} chr{chrom} {method}",
                        flush=True,
                    )
                    continue
                loops = load_loops(tsv)
                detail = score_loops(loops, peaks)
                stats = summarize(detail)
                scored[method] = (loops, detail, stats)

            gt_valid: Set[Loop] = set()
            if "y" in scored:
                gt_valid = gt_valid_set(scored["y"][1])

            for method, (loops, detail, stats) in scored.items():
                n_ov, n_gt, pct_rec = recovery_vs_gt_valid(loops, gt_valid)
                tsv = runs[(sample, chrom, method)]
                row = {
                    "sample": sample,
                    "split": split,
                    "chromosome": chrom_label(chrom),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_loops": stats["n_loops"],
                    "n_both_anchors": stats["n_both_anchors"],
                    "n_single_anchor": stats["n_single_anchor"],
                    "n_no_anchor": stats["n_no_anchor"],
                    "pct_valid_chip": round(stats["pct_valid_chip"], 4),
                    "n_gt_valid": n_gt,
                    "n_overlap_gt_valid": n_ov,
                    "pct_gt_valid_recovered": round(pct_rec, 4),
                    "peak_bed": bed,
                    "tsv": tsv,
                }
                per_chrom_rows.append(row)
                agg_counts[(sample, split, method)].append(
                    (
                        stats["n_both_anchors"],
                        stats["n_loops"],
                        n_ov,
                        n_gt,
                    )
                )
                print(
                    f"{sample} {split} chr{chrom} {METHOD_LABELS[method]}: "
                    f"valid={stats['n_both_anchors']}/{stats['n_loops']} "
                    f"({stats['pct_valid_chip']:.1f}%), "
                    f"GT-valid recovered={n_ov}/{n_gt} ({pct_rec:.1f}%)",
                    flush=True,
                )
                if write_loop_csv and detail:
                    loop_csv = os.path.join(
                        detail_dir,
                        f"{sample}_chr{chrom}_{method}_loops.csv",
                    )
                    with open(loop_csv, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
                        w.writeheader()
                        w.writerows(detail)

    summary_rows: List[Dict] = []
    for sample in samples:
        for split, chroms in chrom_splits.items():
            if not chroms:
                continue
            for method in METHODS:
                pairs = agg_counts.get((sample, split, method), [])
                n_both = sum(a for a, _, _, _ in pairs)
                n_tot = sum(b for _, b, _, _ in pairs)
                n_ov = sum(o for _, _, o, _ in pairs)
                n_gt = sum(g for _, _, _, g in pairs)
                pct = (100.0 * n_both / n_tot) if n_tot else 0.0
                pct_rec = (100.0 * n_ov / n_gt) if n_gt else 0.0
                summary_rows.append(
                    {
                        "sample": sample,
                        "split": split,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "n_loops": n_tot,
                        "n_both_anchors": n_both,
                        "pct_valid_chip": round(pct, 4),
                        "n_gt_valid": n_gt,
                        "n_overlap_gt_valid": n_ov,
                        "pct_gt_valid_recovered": round(pct_rec, 4),
                        "peak_bed": bed,
                        "chromosomes": ",".join(chrom_label(c) for c in chroms),
                    }
                )

    per_chrom_csv = os.path.join(out_dir, "chip_loop_agreement_per_chrom.csv")
    summary_csv = os.path.join(out_dir, "chip_loop_agreement_summary.csv")

    per_fields = [
        "sample",
        "split",
        "chromosome",
        "method",
        "method_label",
        "n_loops",
        "n_both_anchors",
        "n_single_anchor",
        "n_no_anchor",
        "pct_valid_chip",
        "n_gt_valid",
        "n_overlap_gt_valid",
        "pct_gt_valid_recovered",
        "peak_bed",
        "tsv",
    ]
    with open(per_chrom_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=per_fields)
        w.writeheader()
        for row in per_chrom_rows:
            w.writerow(row)

    sum_fields = [
        "sample",
        "split",
        "method",
        "method_label",
        "n_loops",
        "n_both_anchors",
        "pct_valid_chip",
        "n_gt_valid",
        "n_overlap_gt_valid",
        "pct_gt_valid_recovered",
        "peak_bed",
        "chromosomes",
    ]
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    plot_paths = plot_comparison(
        per_chrom_rows,
        out_dir,
        samples=list(samples.keys()),
        chrom_splits=chrom_splits,
    )
    print(f"Wrote {per_chrom_csv}", flush=True)
    print(f"Wrote {summary_csv}", flush=True)
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names = (
        list(DATASETS.keys())
        if args.dataset == "all"
        else [args.dataset]
    )
    for name in names:
        # --output_dir only applies when a single dataset is selected
        out = args.output_dir if len(names) == 1 else None
        run_dataset(
            dataset_name=name,
            mustache_dir=args.mustache_dir,
            peak_bed=args.peak_bed,
            output_dir=out,
            write_loop_csv=args.write_loop_csv,
            script_dir=script_dir,
        )


if __name__ == "__main__":
    main()
