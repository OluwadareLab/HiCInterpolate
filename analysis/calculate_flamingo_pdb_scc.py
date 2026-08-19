#!/usr/bin/env python3
"""Spearman SCC between y and method PDB structures under flamingo_user."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

METHOD_COLS = {
    "pred": "HiCInterpolate",
    "linear": "HL",
    "of": "HOF",
    "4dmax": "4DMax",
}
METHODS = tuple(METHOD_COLS.keys())
_METHOD_SUFFIX_RE = re.compile(
    r"^(?P<out_tag>.+)_(?P<chrom>\d+)_(?P<method>y|pred|4dmax|linear|of)$"
)
_KNOWN_SAMPLE_SUBSAMPLE = (
    ("cerebellar_granule_neuron", "control"),
    ("embryo", "development"),
    ("dmso", "control"),
    ("dtag", "v1"),
)
_ATOM_RE = re.compile(
    r"^ATOM\s+\d+\s+CA\s+\S+\s+(\S+)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def load_coords_pdb(path: str) -> np.ndarray:
    """Load CA atom XYZ from a flamingo PDB (order = genomic order)."""
    coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            m = _ATOM_RE.match(line)
            if m:
                coords.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
                continue
            # fixed-column fallback (standard PDB)
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            if " CA " not in line[12:16] and line[12:16].strip() != "CA":
                continue
            coords.append((x, y, z))
    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 3:
        raise ValueError(f"need >=3 CA atoms in {path}, got {xyz.shape}")
    return xyz


def pairwise_distances(xyz: np.ndarray) -> np.ndarray:
    diff = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def spearman_manual(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def scc_pdb(xyz_a: np.ndarray, xyz_b: np.ndarray) -> Tuple[float, int]:
    """Spearman of upper-triangle pairwise distances (prefix-aligned)."""
    n = min(len(xyz_a), len(xyz_b))
    if n < 3:
        return float("nan"), 0
    da = pairwise_distances(xyz_a[:n])
    db = pairwise_distances(xyz_b[:n])
    iu = np.triu_indices(n, k=1)
    flat_a, flat_b = da[iu], db[iu]
    if spearmanr is not None:
        corr, _ = spearmanr(flat_a, flat_b)
        corr = float(corr)
    else:
        corr = spearman_manual(flat_a, flat_b)
    return corr, n


def pdb_path(root: str, out_tag: str, chrom: str, method: str, start: int, end: int) -> str:
    return os.path.join(
        root,
        f"{out_tag}_{chrom}_{method}",
        f"region_{start}_{end}",
        "flamingo_structure.pdb",
    )


def scc_for_region(
    root: str,
    out_tag: str,
    chrom: str,
    start: int,
    end: int,
    methods: Dict[str, str],
) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {col: None for col in methods.values()}
    scores["n_atoms"] = None
    y_pdb = pdb_path(root, out_tag, chrom, "y", start, end)
    if not os.path.isfile(y_pdb):
        print(f"SKIP missing y pdb: {y_pdb}", flush=True)
        return scores
    try:
        xyz_y = load_coords_pdb(y_pdb)
    except Exception:
        print(f"FAILED load y pdb: {y_pdb}", flush=True)
        traceback.print_exc()
        return scores

    for method, col in methods.items():
        m_pdb = pdb_path(root, out_tag, chrom, method, start, end)
        if not os.path.isfile(m_pdb):
            print(f"SKIP missing {method} pdb: {m_pdb}", flush=True)
            continue
        try:
            xyz_m = load_coords_pdb(m_pdb)
            corr, n = scc_pdb(xyz_y, xyz_m)
            if not np.isfinite(corr):
                print(
                    f"SKIP non-finite SCC {method} chr{chrom} {start}-{end} n={n}",
                    flush=True,
                )
                continue
            scores[col] = corr
            scores["n_atoms"] = n
            print(
                f"chr{chrom} {start}-{end} {col}: SCC={corr:.6f} n={n}",
                flush=True,
            )
        except Exception:
            print(f"FAILED SCC {method} chr{chrom} {start}-{end}", flush=True)
            traceback.print_exc()
    return scores


def _parse_out_tag_meta(out_tag: str) -> Tuple[str, str, str]:
    parts = out_tag.split("_")
    if len(parts) < 5:
        return out_tag, "", ""
    rest = "_".join(parts[2:])
    for sample, subsample in _KNOWN_SAMPLE_SUBSAMPLE:
        prefix = f"{sample}_{subsample}_"
        if rest.startswith(prefix):
            return sample, subsample, rest[len(prefix) :]
    toks = rest.split("_")
    return toks[0], toks[1], "_".join(toks[2:])


def discover_jobs(root: str) -> List[dict]:
    by_key: Dict[Tuple[str, str], dict] = {}
    for name in sorted(os.listdir(root)):
        m = _METHOD_SUFFIX_RE.match(name)
        if not m or m.group("method") != "y":
            continue
        out_tag = m.group("out_tag")
        chrom = m.group("chrom")
        y_dir = os.path.join(root, name)
        regions = []
        for entry in sorted(os.listdir(y_dir)):
            if not entry.startswith("region_"):
                continue
            pdb = os.path.join(y_dir, entry, "flamingo_structure.pdb")
            if not os.path.isfile(pdb):
                continue
            try:
                _, start_s, end_s = entry.split("_", 2)
                regions.append((int(start_s), int(end_s)))
            except ValueError:
                continue
        if not regions:
            continue
        sample, subsample, timestamp = _parse_out_tag_meta(out_tag)
        by_key[(out_tag, chrom)] = {
            "sample": sample,
            "subsample": subsample,
            "timestamp": timestamp,
            "chromosome": chrom,
            "out_tag": out_tag,
            "regions": regions,
        }
    return list(by_key.values())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flamingo_root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/flamingo_user",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
        help="Methods to compare against y (default: all).",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Default: <flamingo_root>/flamingo_pdb_scc.csv",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    methods = {m: METHOD_COLS[m] for m in args.methods}
    fieldnames = [
        "sample",
        "subsample",
        "timestamp",
        "chromosome",
        "region",
        *[methods[m] for m in methods],
        "n_atoms",
    ]

    jobs = discover_jobs(args.flamingo_root)
    print(f"Discovered {len(jobs)} jobs under {args.flamingo_root}", flush=True)

    out_csv = args.output_csv or os.path.join(args.flamingo_root, "flamingo_pdb_scc.csv")
    rows = []
    for job in jobs:
        for start, end in job["regions"]:
            scores = scc_for_region(
                args.flamingo_root,
                job["out_tag"],
                job["chromosome"],
                start,
                end,
                methods,
            )
            rows.append(
                {
                    "sample": job["sample"],
                    "subsample": job["subsample"],
                    "timestamp": job["timestamp"],
                    "chromosome": job["chromosome"],
                    "region": f"{start}-{end}",
                    **{c: scores.get(c) for c in methods.values()},
                    "n_atoms": scores.get("n_atoms"),
                }
            )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_ours = sum(1 for r in rows if r.get("HiCInterpolate") is not None)
    print(f"Wrote {out_csv} ({len(rows)} rows, {n_ours} with ours/pred)", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
