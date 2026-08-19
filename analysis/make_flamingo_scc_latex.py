#!/usr/bin/env python3
"""Build LaTeX table from flamingo SCC CSV; bold max score per row."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

METHOD_COLS = ("HiCInterpolate", "HL", "HOF", "4DMax")
METHOD_HEADERS = ("HiCInterpolate", "HL", "HOF", "4DMax")

SAMPLE_LABELS = {
    ("dmso", "control"): "Human DMSO",
    ("dtag", "v1"): "Human dTAG",
    ("cerebellar_granule_neuron", "control"): "Mouse CGN",
    ("embryo", "development"): "Mouse embryo",
}

TIMESTAMP_LABELS = {
    "early2_cell": "early 2-cell",
    "late2_cell": "late 2-cell",
    "8cell": "8-cell",
}

DEFAULT_CSV = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/flamingo_user/"
    "flamingo_pdb_scc.csv"
)


def _f(v: Optional[str]) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def load_rows(path: str) -> List[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # normalize method key casing
    out = []
    for r in rows:
        nr = dict(r)
        aliases = {
            "HiCInterpolate": ("HiCInterpolate", "Ours", "ours"),
            "HL": ("HL", "Linear"),
            "HOF": ("HOF", "Optical Flow"),
            "4DMax": ("4DMax",),
        }
        for dest, keys in aliases.items():
            if dest not in nr or nr[dest] in (None, ""):
                for k in keys:
                    if k in nr and nr[k] not in (None, ""):
                        nr[dest] = nr[k]
                        break
        out.append(nr)
    return out


def row_label(r: dict) -> str:
    key = (r.get("sample", ""), r.get("subsample", ""))
    base = SAMPLE_LABELS.get(key, r.get("sample", ""))
    ts = r.get("timestamp", "")
    if key == ("embryo", "development"):
        return f"{base} ({TIMESTAMP_LABELS.get(ts, ts)})"
    return base


def fmt_score(v: Optional[float], bold: bool, digits: int = 3) -> str:
    if v is None:
        return "---"
    s = f"{v:.{digits}f}"
    return f"\\textbf{{{s}}}" if bold else s


def scores_for_row(r: dict) -> List[Optional[float]]:
    return [_f(r.get(c)) for c in METHOD_COLS]


def bold_mask(vals: Sequence[Optional[float]]) -> List[bool]:
    finite = [v for v in vals if v is not None]
    if not finite:
        return [False] * len(vals)
    best = max(finite)
    return [v is not None and abs(v - best) < 1e-12 for v in vals]


def sort_key(r: dict) -> Tuple:
    sample_order = {
        "dmso": 0,
        "dtag": 1,
        "cerebellar_granule_neuron": 2,
        "embryo": 3,
    }
    ts_order = {"early2_cell": 0, "late2_cell": 1, "8cell": 2}
    return (
        sample_order.get(r.get("sample", ""), 99),
        ts_order.get(r.get("timestamp", ""), 99),
        int(r.get("chromosome", 0)),
    )


def build_table(
    rows: List[dict],
    caption: str,
    label: str,
    digits: int = 3,
) -> str:
    rows = sorted(rows, key=sort_key)
    lines = [
        r"% Auto-generated SCC table; highest score per row in bold.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{ll" + "c" * len(METHOD_HEADERS) + "}",
        r"\toprule",
        "Sample & Chr & " + " & ".join(METHOD_HEADERS) + r" \\",
        r"\midrule",
    ]

    prev_group = None
    for r in rows:
        label_s = row_label(r)
        chrom = str(r.get("chromosome", "")).strip()
        if not chrom.lower().startswith("chr"):
            chrom = f"chr{chrom}"
        vals = scores_for_row(r)
        mask = bold_mask(vals)
        cells = [fmt_score(v, b, digits) for v, b in zip(vals, mask)]

        group = (r.get("sample"), r.get("timestamp"))
        if prev_group is not None and group != prev_group:
            lines.append(r"\addlinespace")
        prev_group = group

        lines.append(f"{label_s} & {chrom} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument(
        "--output",
        default=None,
        help="Default: <csv_dir>/flamingo_pdb_scc_table.tex",
    )
    p.add_argument("--digits", type=int, default=3)
    p.add_argument(
        "--caption",
        default=(
            "Spearman correlation coefficient (SCC) between ground-truth ($y$) "
            "and reconstructed 3D structures. Highest score per row in bold."
        ),
    )
    p.add_argument("--label", default="tab:flamingo_pdb_scc")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args.csv)
    tex = build_table(rows, args.caption, args.label, digits=args.digits)
    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)),
        "flamingo_pdb_scc_table.tex",
    )
    with open(out, "w") as f:
        f.write(tex)
    print(tex)
    print(f"% Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
