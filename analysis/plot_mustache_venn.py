#!/usr/bin/env python3
"""Draw Mustache loop overlap Venn diagrams: y vs pred / linear / of."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

METHODS = ("pred", "linear", "of", "4dmax")
METHOD_LABELS = {
    "pred": "Ours",
    "linear": "Linear",
    "of": "Optical Flow",
    "4dmax": "4DMax",
}
GT_LABEL = "Ground Truth"
NATURE_COLORS = ["#009e74", "#0072b2", "#d55e00", "#000000"]
DIR_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<method>y|pred|linear|of|4dmax)$"
)


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def load_loop_set(tsv_path: str) -> Set[Tuple[str, int, int, str, int, int]]:
    if not os.path.isfile(tsv_path) or os.path.getsize(tsv_path) == 0:
        return set()
    df = pd.read_csv(tsv_path, sep="\t")
    required = [
        "BIN1_CHR", "BIN1_START", "BIN1_END",
        "BIN2_CHROMOSOME", "BIN2_START", "BIN2_END",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{tsv_path}: missing columns {missing}")
    loops = set()
    for row in df.itertuples(index=False):
        loops.add(
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
    candidates = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.endswith(".mustache.tsv")
    ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def discover_groups(mustache_root: str) -> Dict[str, Dict[str, str]]:
    """prefix -> {method: tsv_path}."""
    groups: Dict[str, Dict[str, str]] = defaultdict(dict)
    for name in sorted(os.listdir(mustache_root)):
        path = os.path.join(mustache_root, name)
        if not os.path.isdir(path):
            continue
        m = DIR_RE.match(name)
        if not m:
            continue
        tsv = find_mustache_tsv(path)
        if tsv is None:
            continue
        groups[m.group("prefix")][m.group("method")] = tsv
    return groups


def draw_venn2(
    set_a: Set,
    set_b: Set,
    label_a: str,
    label_b: str,
    output_file: str,
) -> dict:
    inter = set_a & set_b
    only_a = len(set_a) - len(inter)
    only_b = len(set_b) - len(inter)
    n_inter = len(inter)
    recovery = (100.0 * n_inter / len(set_a)) if set_a else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    c0, c1 = NATURE_COLORS[0], NATURE_COLORS[1]
    ax.add_patch(Circle((0.38, 0.5), 0.28, alpha=0.45, color=c0, lw=0))
    ax.add_patch(Circle((0.62, 0.5), 0.28, alpha=0.45, color=c1, lw=0))

    ax.text(0.22, 0.5, str(only_a), fontsize=14, ha="center", va="center", color=NATURE_COLORS[3])
    ax.text(0.78, 0.5, str(only_b), fontsize=14, ha="center", va="center", color=NATURE_COLORS[3])
    ax.text(0.50, 0.5, str(n_inter), fontsize=16, ha="center", va="center",
            color=NATURE_COLORS[3], fontweight="bold")

    ax.text(0.18, 0.88, label_a, fontsize=12, ha="center", va="center", color=c0)
    ax.text(0.82, 0.88, label_b, fontsize=12, ha="center", va="center", color=c1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "only_y": only_a,
        "only_method": only_b,
        "overlap": n_inter,
        "n_y": len(set_a),
        "n_method": len(set_b),
        "recovery_rate": recovery,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Mustache loop Venn: y vs pred, y vs linear, y vs of per dataset."
    )
    p.add_argument(
        "--mustache_dir",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/mustache",
        help="Root directory of Mustache run folders",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <mustache_dir>/venn)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mustache_dir = args.mustache_dir
    output_dir = args.output_dir or os.path.join(mustache_dir, "venn")
    os.makedirs(output_dir, exist_ok=True)

    groups = discover_groups(mustache_dir)
    if not groups:
        raise SystemExit(f"No Mustache groups found under {mustache_dir}")

    rows = []
    n_plots = 0

    for prefix, paths in sorted(groups.items()):
        if "y" not in paths:
            continue
        y_loops = load_loop_set(paths["y"])
        group_out = os.path.join(output_dir, prefix)
        os.makedirs(group_out, exist_ok=True)

        for method in METHODS:
            if method not in paths:
                print(f"SKIP {prefix}: missing {method}", flush=True)
                continue
            m_loops = load_loop_set(paths[method])
            method_label = METHOD_LABELS[method]
            slug = method_label.lower().replace(" ", "_")
            out = os.path.join(group_out, f"ground_truth_vs_{slug}.png")
            stats = draw_venn2(y_loops, m_loops, GT_LABEL, method_label, out)
            rows.append({"group": prefix, "method": method, **stats, "plot": out})
            n_plots += 1
            print(
                f"{prefix} {GT_LABEL} vs {method_label}: overlap={stats['overlap']} "
                f"recovery={stats['recovery_rate']:.1f}% -> {out}",
                flush=True,
            )

    csv_path = os.path.join(output_dir, "venn_summary.csv")
    fieldnames = [
        "group", "method", "n_y", "n_method", "only_y", "only_method",
        "overlap", "recovery_rate", "plot",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {n_plots} plots. Summary: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
