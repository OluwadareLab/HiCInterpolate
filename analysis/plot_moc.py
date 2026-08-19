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

METHOD_COLS = ("HiCInterpolate", "HL", "HOF", "4DMax")
METHOD_COLORS = {
    "HiCInterpolate": "#009E73",
    "HL": "#0072B2",
    "HOF": "#E69F00",
    "4DMax": "#CC79A7",
}
_METHOD_CSV_KEYS = {
    "HiCInterpolate": ("HiCInterpolate", "Ours", "ours"),
    "HL": ("HL", "Linear"),
    "HOF": ("HOF", "Optical Flow"),
    "4DMax": ("4DMax",),
}
TOOLS = ("embedtad", "topdom", "spectral")
TOOL_LABELS = {
    "embedtad": "EmbedTAD",
    "topdom": "TopDom",
    "spectral": "Spectral"
}

TIMESTAMP_ORDER = ("early2_cell", "late2_cell", "8cell")
TIMESTAMP_LABELS = {
    "early2_cell": "Early 2-cell",
    "late2_cell": "Late 2-cell",
    "8cell": "8-cell"
}

DEFAULT_MOC_CSV = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/tads/moc.csv"
)

CELL_CYCLE_CHROMS = (10, 15, 19)


def chrom_label(chrom) -> str:
    c = str(chrom).strip()
    return c if c.lower().startswith("chr") else f"chr{c}"


PLOT_SPECS = [
    {
        "file_stem": "moc_dmso_seen",
        "dataset": "human",
        "sample": "dmso",
        "chroms": (11, 16, 21),
        "mode": "chrom",
    },
    {
        "file_stem": "moc_dmso_unseen",
        "dataset": "human",
        "sample": "dmso",
        "chroms": (10, 15, 20),
        "mode": "chrom",
    },
    {
        "file_stem": "moc_dtag_seen",
        "dataset": "human",
        "sample": "dtag",
        "chroms": (11, 16, 21),
        "mode": "chrom",
    },
    {
        "file_stem": "moc_dtag_unseen",
        "dataset": "human",
        "sample": "dtag",
        "chroms": (10, 15, 20),
        "mode": "chrom",
    },
    {
        "file_stem": "moc_cerebellar_cross_organism",
        "dataset": "mouse",
        "sample": "cerebellar_granule_neuron",
        "chroms": (10, 15, 19),
        "mode": "chrom",
    },
] + [
    {
        "file_stem": f"moc_embryo_cell_cycle_{chrom_label(c)}",
        "dataset": "mouse",
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


def _row_method_value(r: Dict, method: str) -> Optional[float]:
    for key in _METHOD_CSV_KEYS[method]:
        if key in r:
            return _to_float(r.get(key))
    return None


def load_moc_csv(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            entry = {
                "dataset": r["dataset"],
                "sample": r["sample"],
                "subsample": r["subsample"],
                "timestamp": r["timestamp"],
                "chromosome": int(r["chromosome"]),
                "tool": r["tool"],
            }
            for method in METHOD_COLS:
                entry[method] = _row_method_value(r, method)
            rows.append(entry)
    return rows


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def filter_rows(
    rows: Sequence[Dict],
    dataset: str,
    sample: str,
    chroms: Sequence[int],
) -> List[Dict]:
    chrom_set = {int(c) for c in chroms}
    return [
        r
        for r in rows
        if r["dataset"] == dataset
        and r["sample"] == sample
        and r["chromosome"] in chrom_set
    ]


def values_by_chrom(
    rows: Sequence[Dict], tool: str, chroms: Sequence[int]
) -> Dict[str, List[Optional[float]]]:
    lookup = {
        (r["chromosome"], r["tool"]): r
        for r in rows
        if r["tool"] == tool
    }
    out: Dict[str, List[Optional[float]]] = {m: [] for m in METHOD_COLS}
    for chrom in chroms:
        r = lookup.get((int(chrom), tool))
        for m in METHOD_COLS:
            out[m].append(None if r is None else r[m])
    return out


def values_by_timestamp(
    rows: Sequence[Dict],
    tool: str,
    chrom: int,
    timestamps: Sequence[str],
) -> Dict[str, List[Optional[float]]]:
    lookup = {
        (r["timestamp"], r["tool"]): r
        for r in rows
        if r["tool"] == tool and r["chromosome"] == int(chrom)
    }
    out: Dict[str, List[Optional[float]]] = {m: [] for m in METHOD_COLS}
    for ts in timestamps:
        r = lookup.get((ts, tool))
        for m in METHOD_COLS:
            out[m].append(None if r is None else r[m])
    return out


def _plot_grouped_bars(
    ax,
    x_labels: Sequence[str],
    method_values: Dict[str, List[Optional[float]]],
    ylabel: str,
    show_ylabel: bool,
    panel_label: str,
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
            if raw_v is None or val <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
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
    ax.set_ylim(0, max(100.0, ymax) * 1.18 if ymax > 0 else 108)
    ax.set_ylabel(ylabel if show_ylabel else "")
    ax.set_title(panel_label, pad=1, fontsize=8, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles


def draw_panel(
    rows: Sequence[Dict],
    spec: Dict,
    output_path: str,
) -> Optional[str]:
    subset = filter_rows(
        rows, spec["dataset"], spec["sample"], spec["chroms"]
    )
    if not subset:
        print(f"SKIP empty: {spec['file_stem']}", flush=True)
        return None

    apply_nature_style()
    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(
        2,
        len(TOOLS),
        height_ratios=[0.07, 1.0],
        hspace=0.08,
        wspace=0.22,
        left=0.07,
        right=0.99,
        top=0.97,
        bottom=0.14,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    axes = [fig.add_subplot(gs[1, 0])]
    for i in range(1, len(TOOLS)):
        axes.append(fig.add_subplot(gs[1, i], sharey=axes[0]))

    handles = []
    for ax_i, (ax, tool) in enumerate(zip(axes, TOOLS)):
        if spec["mode"] == "timestamp":
            chrom = int(spec["chroms"][0])
            present = [
                ts
                for ts in TIMESTAMP_ORDER
                if any(
                    r["timestamp"] == ts and r["chromosome"] == chrom
                    for r in subset
                )
            ]
            if not present:
                present = sorted(
                    {
                        r["timestamp"]
                        for r in subset
                        if r["chromosome"] == chrom
                    }
                )
            x_labels = [TIMESTAMP_LABELS.get(ts, ts) for ts in present]
            method_values = values_by_timestamp(
                subset, tool, chrom, present
            )
        else:
            x_labels = [chrom_label(c) for c in spec["chroms"]]
            method_values = values_by_chrom(subset, tool, spec["chroms"])

        h = _plot_grouped_bars(
            ax,
            x_labels=x_labels,
            method_values=method_values,
            ylabel="MoC (%)",
            show_ylabel=(ax_i == 0),
            panel_label=TOOL_LABELS[tool],
        )
        if ax_i == 0:
            handles = h
        if ax_i > 0:
            ax.tick_params(labelleft=False)

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
    p.add_argument("--moc_csv", default=DEFAULT_MOC_CSV)
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: <dirname(moc_csv)>/moc_plots",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.moc_csv):
        raise SystemExit(f"Missing MoC CSV: {args.moc_csv}")

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.moc_csv)), "moc_plots"
    )
    rows = load_moc_csv(args.moc_csv)
    written: List[str] = []
    for spec in PLOT_SPECS:
        path = os.path.join(out_dir, f"{spec['file_stem']}.png")
        out = draw_panel(rows, spec, path)
        if out:
            written.append(out)

    old = os.path.join(out_dir, "moc_embryo_cell_cycle.png")
    if os.path.isfile(old):
        os.remove(old)
        print(f"Removed {old}", flush=True)

    print(f"Done ({len(written)} figures) -> {out_dir}", flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
