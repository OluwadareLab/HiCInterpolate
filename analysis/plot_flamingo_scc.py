#!/usr/bin/env python3
"""Nature-style SCC bar plots from flamingo_scc CSV.

Reads calculate_flamingo_scc.py output and writes one 300 dpi PNG per panel:

  Human DMSO / dTAG:
    seen   — chr11, 16, 21
    unseen — chr10, 15, 20
  Mouse cerebellar granule neuron — cross-organism (all chroms)
  Mouse embryo development — cell cycle (one figure per chromosome)

Grouped bars: Ours / 4DMax / Linear / Optical Flow. Legend top, 4 columns.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DPI = 300

METHOD_COLS = ("Ours", "4DMax", "Linear", "Optical Flow")
METHOD_COLORS = {
    "Ours": "#009E73",
    "4DMax": "#CC79A7",
    "Linear": "#0072B2",
    "Optical Flow": "#E69F00",
}
# CSV may use "ours" (calculate_flamingo_scc) or "Ours"
_METHOD_CSV_KEYS = {
    "Ours": ("Ours", "ours"),
    "4DMax": ("4DMax",),
    "Linear": ("Linear",),
    "Optical Flow": ("Optical Flow",),
}

TIMESTAMP_ORDER = ("early2_cell", "late2_cell", "8cell")
TIMESTAMP_LABELS = {
    "early2_cell": "Early 2-cell",
    "late2_cell": "Late 2-cell",
    "8cell": "8-cell",
}

DEFAULT_SCC_CSV = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/flamingo_user/"
    "flamingo_scc_2500_4000.csv"
)

CELL_CYCLE_CHROMS = (10, 15, 19)

HUMAN_SAMPLES = frozenset({"dmso", "dtag"})
MOUSE_SAMPLES = frozenset({"cerebellar_granule_neuron", "embryo"})


def chrom_label(chrom) -> str:
    c = str(chrom).strip()
    return c if c.lower().startswith("chr") else f"chr{c}"


PLOT_SPECS = [
    {
        "file_stem": "scc_dmso_seen",
        "sample": "dmso",
        "chroms": (11, 16, 21),
        "mode": "chrom",
    },
    {
        "file_stem": "scc_dmso_unseen",
        "sample": "dmso",
        "chroms": (10, 15, 20),
        "mode": "chrom",
    },
    {
        "file_stem": "scc_dtag_seen",
        "sample": "dtag",
        "chroms": (11, 16, 21),
        "mode": "chrom",
    },
    {
        "file_stem": "scc_dtag_unseen",
        "sample": "dtag",
        "chroms": (10, 15, 20),
        "mode": "chrom",
    },
    {
        "file_stem": "scc_cerebellar_cross_organism",
        "sample": "cerebellar_granule_neuron",
        "chroms": (10, 15, 19),
        "mode": "chrom",
    },
] + [
    {
        "file_stem": f"scc_embryo_cell_cycle_{chrom_label(c)}",
        "sample": "embryo",
        "chroms": (c,),
        "mode": "timestamp",
    }
    for c in CELL_CYCLE_CHROMS
]


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "font.weight": "medium",
            "axes.labelsize": 8,
            "axes.labelweight": "medium",
            "axes.titlesize": 8,
            "axes.titleweight": "medium",
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


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_method_value(r: Dict, method: str) -> Optional[float]:
    for key in _METHOD_CSV_KEYS[method]:
        if key in r and r[key] is not None:
            return r[key]
    return None


def load_scc_csv(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sample = r["sample"]
            if sample in HUMAN_SAMPLES:
                dataset = "human"
            elif sample in MOUSE_SAMPLES:
                dataset = "mouse"
            else:
                dataset = "unknown"
            entry = {
                "dataset": dataset,
                "sample": sample,
                "subsample": r.get("subsample", ""),
                "timestamp": r.get("timestamp", ""),
                "chromosome": int(r["chromosome"]),
                "region": r.get("region", ""),
            }
            for method in METHOD_COLS:
                val = None
                for key in _METHOD_CSV_KEYS[method]:
                    if key in r:
                        val = _to_float(r.get(key))
                        break
                entry[method] = val
            rows.append(entry)
    return rows


def filter_rows(
    rows: Sequence[Dict],
    sample: str,
    chroms: Sequence[int],
) -> List[Dict]:
    chrom_set = {int(c) for c in chroms}
    return [
        r
        for r in rows
        if r["sample"] == sample and r["chromosome"] in chrom_set
    ]


def values_by_chrom(
    rows: Sequence[Dict], chroms: Sequence[int]
) -> Dict[str, List[Optional[float]]]:
    lookup = {r["chromosome"]: r for r in rows}
    out: Dict[str, List[Optional[float]]] = {m: [] for m in METHOD_COLS}
    for chrom in chroms:
        r = lookup.get(int(chrom))
        for m in METHOD_COLS:
            out[m].append(None if r is None else _row_method_value(r, m))
    return out


def values_by_timestamp(
    rows: Sequence[Dict],
    chrom: int,
    timestamps: Sequence[str],
) -> Dict[str, List[Optional[float]]]:
    lookup = {
        r["timestamp"]: r
        for r in rows
        if r["chromosome"] == int(chrom)
    }
    out: Dict[str, List[Optional[float]]] = {m: [] for m in METHOD_COLS}
    for ts in timestamps:
        r = lookup.get(ts)
        for m in METHOD_COLS:
            out[m].append(None if r is None else _row_method_value(r, m))
    return out


def _plot_grouped_bars(
    ax,
    x_labels: Sequence[str],
    method_values: Dict[str, List[Optional[float]]],
    ylabel: str,
) -> List:
    x = np.arange(len(x_labels), dtype=float)
    n_m = len(METHOD_COLS)
    width = min(0.18, 0.78 / max(n_m, 1))
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    ymax = 0.0
    handles = []

    for i, method in enumerate(METHOD_COLS):
        raw = method_values.get(method, [None] * len(x_labels))
        vals = [0.0 if v is None else float(v) for v in raw]
        ymax = max(ymax, max(vals) if vals else 0.0)
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            label=method,
            alpha=0.92,
        )
        handles.append(bars)
        for bar, val, raw_v in zip(bars, vals, raw):
            if raw_v is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.0,
                fontweight="medium",
                color="black",
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(x_labels))
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, ymax) * 1.18 if ymax > 0 else 1.05)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles


def draw_panel(
    rows: Sequence[Dict],
    spec: Dict,
    output_path: str,
) -> Optional[str]:
    subset = filter_rows(rows, spec["sample"], spec["chroms"])
    # Drop chroms with no rows so empty slots (e.g. chr21) do not pad the axis
    if spec["mode"] == "chrom":
        present_chroms = [
            c for c in spec["chroms"] if any(r["chromosome"] == int(c) for r in subset)
        ]
        if not present_chroms:
            print(f"SKIP empty: {spec['file_stem']}", flush=True)
            return None
    else:
        present_chroms = list(spec["chroms"])
        if not subset:
            print(f"SKIP empty: {spec['file_stem']}", flush=True)
            return None

    apply_nature_style()
    fig = plt.figure(figsize=(4.8, 2.6))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.08, 1.0],
        hspace=0.05,
        left=0.12,
        right=0.98,
        top=0.96,
        bottom=0.16,
    )
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.axis("off")
    ax = fig.add_subplot(gs[1, 0])

    if spec["mode"] == "timestamp":
        chrom = int(spec["chroms"][0])
        present = [
            ts
            for ts in TIMESTAMP_ORDER
            if any(r["timestamp"] == ts and r["chromosome"] == chrom for r in subset)
        ]
        if not present:
            present = sorted(
                {r["timestamp"] for r in subset if r["chromosome"] == chrom}
            )
        x_labels = [TIMESTAMP_LABELS.get(ts, ts) for ts in present]
        method_values = values_by_timestamp(subset, chrom, present)
    else:
        x_labels = [chrom_label(c) for c in present_chroms]
        method_values = values_by_chrom(subset, present_chroms)

    handles = _plot_grouped_bars(
        ax,
        x_labels=x_labels,
        method_values=method_values,
        ylabel="SCC",
    )
    ax_leg.legend(
        handles,
        list(METHOD_COLS),
        loc="center",
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {output_path}", flush=True)
    return output_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scc_csv", default=DEFAULT_SCC_CSV)
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: <dirname(scc_csv)>/scc_plots",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.scc_csv):
        raise SystemExit(f"Missing SCC CSV: {args.scc_csv}")

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.scc_csv)), "scc_plots"
    )
    rows = load_scc_csv(args.scc_csv)
    written: List[str] = []
    for spec in PLOT_SPECS:
        path = os.path.join(out_dir, f"{spec['file_stem']}.png")
        out = draw_panel(rows, spec, path)
        if out:
            written.append(out)

    print(f"Done ({len(written)} figures) -> {out_dir}", flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
