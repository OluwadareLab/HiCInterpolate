#!/usr/bin/env python3
"""MoC of TAD calls vs ground-truth y for all datasets (run_mustache layout).

For each TAD caller (embedtad / topdom / spectral), compares pred / 4dmax /
linear / of against y of the same caller.

CSV columns:
  dataset, sample, subsample, timestamp, chromosome, tool,
  Ours, 4DMax, Linear, Optical Flow
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

import tad_scores

DEFAULT_TADS_ROOT = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/tads"
)

METHODS = ("pred", "4dmax", "linear", "of")
METHOD_COLS = {
    "pred": "Ours",
    "4dmax": "4DMax",
    "linear": "Linear",
    "of": "Optical Flow",
}
TOOLS = ("embedtad", "topdom", "spectral")

# (organism, sample, subsample) — timestamp is the remainder of the body.
SAMPLE_KEYS = [
    ("human", "dmso", "control"),
    ("human", "dtag", "v1"),
    ("mouse", "cerebellar_granule_neuron", "control"),
    ("mouse", "embryo", "development"),
]

DIR_RE = re.compile(
    r"^(?P<res>\d+)_(?P<organism>human|mouse)_(?P<body>.+)_(?P<chrom>\d+)_"
    r"(?P<method>y|pred|linear|of|4dmax)$"
)

FIELDNAMES = [
    "dataset",
    "sample",
    "subsample",
    "timestamp",
    "chromosome",
    "tool",
    "Ours",
    "4DMax",
    "Linear",
    "Optical Flow",
]


def parse_body(
    organism: str, body: str
) -> Optional[Tuple[str, str, str]]:
    """Return (sample, subsample, timestamp) for a known sample key."""
    candidates = [
        (sample, subsample)
        for org, sample, subsample in SAMPLE_KEYS
        if org == organism
    ]
    candidates.sort(key=lambda x: len(f"{x[0]}_{x[1]}"), reverse=True)
    for sample, subsample in candidates:
        prefix = f"{sample}_{subsample}_"
        if body.startswith(prefix):
            timestamp = body[len(prefix) :]
            if timestamp:
                return sample, subsample, timestamp
    return None


def load_tads(bed_path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(bed_path) or os.path.getsize(bed_path) == 0:
        return None
    df = pd.read_csv(bed_path, sep="\t", header=None)
    if df.shape[1] < 2:
        return None
    out = df.iloc[:, :2].copy()
    out.columns = [0, 1]
    out[0] = out[0].astype(int)
    out[1] = out[1].astype(int)
    return out


def discover_groups(
    tads_root: str,
) -> Dict[Tuple[str, str, str, str, str], Dict[str, str]]:
    """(dataset, sample, subsample, timestamp, chrom) -> {method: run_dir}."""
    groups: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = defaultdict(
        dict
    )
    for name in sorted(os.listdir(tads_root)):
        path = os.path.join(tads_root, name)
        if not os.path.isdir(path):
            continue
        m = DIR_RE.match(name)
        if not m:
            continue
        organism = m.group("organism")
        parsed = parse_body(organism, m.group("body"))
        if parsed is None:
            print(f"SKIP unknown body: {name}", flush=True)
            continue
        sample, subsample, timestamp = parsed
        key = (organism, sample, subsample, timestamp, m.group("chrom"))
        groups[key][m.group("method")] = path
    return groups


def moc_vs_y(
    y_tads: pd.DataFrame, method_tads: pd.DataFrame
) -> float:
    return float(tad_scores.get_moc(tads=method_tads, true_tads=y_tads))


def score_group(
    method_dirs: Dict[str, str], tool: str
) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {col: None for col in METHOD_COLS.values()}
    y_dir = method_dirs.get("y")
    if y_dir is None:
        return scores
    y_bed = os.path.join(y_dir, f"{tool}.bed")
    y_tads = load_tads(y_bed)
    if y_tads is None:
        print(f"SKIP missing/empty y: {y_bed}", flush=True)
        return scores

    for method, col in METHOD_COLS.items():
        m_dir = method_dirs.get(method)
        if m_dir is None:
            continue
        m_bed = os.path.join(m_dir, f"{tool}.bed")
        m_tads = load_tads(m_bed)
        if m_tads is None:
            print(f"SKIP missing/empty {method}: {m_bed}", flush=True)
            continue
        try:
            scores[col] = moc_vs_y(y_tads, m_tads)
        except Exception as exc:
            print(f"FAILED MoC {tool} {method} ({m_bed}): {exc}", flush=True)
    return scores


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tads_root", default=DEFAULT_TADS_ROOT)
    p.add_argument(
        "--tools",
        nargs="+",
        default=list(TOOLS),
        choices=list(TOOLS),
        help="TAD callers to score (default: all)",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Default: <tads_root>/moc.csv",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_csv = args.output_csv or os.path.join(args.tads_root, "moc.csv")
    groups = discover_groups(args.tads_root)

    rows: List[dict] = []
    for (dataset, sample, subsample, timestamp, chrom) in sorted(groups):
        method_dirs = groups[(dataset, sample, subsample, timestamp, chrom)]
        if "y" not in method_dirs:
            print(
                f"SKIP no y for {dataset}/{sample}/{subsample}/{timestamp} chr{chrom}",
                flush=True,
            )
            continue
        for tool in args.tools:
            scores = score_group(method_dirs, tool)
            print(
                f"{dataset} {sample}/{subsample}/{timestamp} chr{chrom} {tool}: "
                f"{scores}",
                flush=True,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "subsample": subsample,
                    "timestamp": timestamp,
                    "chromosome": chrom,
                    "tool": tool,
                    **scores,
                }
            )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
