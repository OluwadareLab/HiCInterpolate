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
    "pred": "ours",
    "4dmax": "4DMax",
    "linear": "Linear",
    "of": "Optical Flow",
}
METHODS = tuple(METHOD_COLS.keys())
FIELDNAMES = [
    "sample",
    "subsample",
    "timestamp",
    "chromosome",
    "region",
    "ours",
    "4DMax",
    "Linear",
    "Optical Flow",
]
_METHOD_SUFFIX_RE = re.compile(
    r"^(?P<out_tag>.+)_(?P<chrom>\d+)_(?P<method>y|pred|4dmax|linear|of)$"
)
# (sample, subsample) — longest sample names first for out_tag parsing
_KNOWN_SAMPLE_SUBSAMPLE = (
    ("cerebellar_granule_neuron", "control"),
    ("embryo", "development"),
    ("dmso", "control"),
    ("dtag", "v1"),
)


def load_coords_tsv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    names = [n.lower() for n in data.dtype.names]
    if "frag_id" in names:
        frag = np.asarray(data[data.dtype.names[names.index("frag_id")]], dtype=np.int64)
    else:
        frag = np.asarray(data[data.dtype.names[0]], dtype=np.int64)
    xyz = np.column_stack(
        [data[data.dtype.names[names.index(k)]] for k in ("x", "y", "z")]
    ).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1)
    return frag[ok], xyz[ok]


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


def scc_aligned(frag_a, xyz_a, frag_b, xyz_b) -> Tuple[float, int]:
    """Spearman of upper-triangle pairwise distances on shared frag_ids."""
    common = np.intersect1d(frag_a, frag_b)
    if common.size < 3:
        return float("nan"), 0
    ia = {int(f): i for i, f in enumerate(frag_a)}
    ib = {int(f): i for i, f in enumerate(frag_b)}
    idx_a = np.array([ia[int(f)] for f in common], dtype=np.int64)
    idx_b = np.array([ib[int(f)] for f in common], dtype=np.int64)
    da = pairwise_distances(xyz_a[idx_a])
    db = pairwise_distances(xyz_b[idx_b])
    iu = np.triu_indices(common.size, k=1)
    flat_a = da[iu]
    flat_b = db[iu]
    # if spearmanr is not None:
    corr, _ = spearmanr(flat_a, flat_b)
    corr = float(corr)
    # else:
    #     corr = spearman_manual(flat_a, flat_b)
    return corr, int(common.size)


def parse_region(s: str) -> Tuple[int, int]:
    s = s.strip()
    if ":" in s:
        a, b = s.split(":", 1)
    elif "-" in s:
        a, b = s.split("-", 1)
    else:
        raise ValueError(f"region must be START:END, got {s!r}")
    start, end = int(a), int(b)
    if end <= start:
        raise ValueError(f"region end must be > start: {s!r}")
    return start, end


def coords_path(root: str, out_tag: str, chrom: str, method: str, start: int, end: int) -> str:
    return os.path.join(
        root,
        f"{out_tag}_{chrom}_{method}",
        f"region_{start}_{end}",
        "flamingo_coords.tsv",
    )


def scc_for_region(
    root: str,
    out_tag: str,
    chrom: str,
    start: int,
    end: int,
) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {col: None for col in METHOD_COLS.values()}
    y_tsv = coords_path(root, out_tag, chrom, "y", start, end)
    if not os.path.isfile(y_tsv):
        print(f"SKIP missing y coords: {y_tsv}", flush=True)
        return scores

    try:
        frag_y, xyz_y = load_coords_tsv(y_tsv)
    except Exception:
        print(f"FAILED load y coords: {y_tsv}", flush=True)
        traceback.print_exc()
        return scores

    for method, col in METHOD_COLS.items():
        m_tsv = coords_path(root, out_tag, chrom, method, start, end)
        if not os.path.isfile(m_tsv):
            print(f"SKIP missing {method} coords: {m_tsv}", flush=True)
            continue
        try:
            frag_m, xyz_m = load_coords_tsv(m_tsv)
            corr, n_common = scc_aligned(frag_y, xyz_y, frag_m, xyz_m)
            if not np.isfinite(corr):
                print(
                    f"SKIP non-finite SCC {method} chr{chrom} {start}-{end} n={n_common}",
                    flush=True,
                )
                continue
            scores[col] = float(corr)
            print(
                f"chr{chrom} {start}-{end} {col}: SCC={corr:.6f} n={n_common}",
                flush=True,
            )
        except Exception:
            print(f"FAILED SCC {method} chr{chrom} {start}-{end}", flush=True)
            traceback.print_exc()
    return scores


def load_jobs(path: str) -> List[dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 6:
                raise ValueError(f"bad jobs line (want 6 tab fields): {line!r}")
            sample, subsample, timestamp, chrom, out_tag, regions_csv = parts
            regions = [parse_region(r) for r in regions_csv.split(",") if r.strip()]
            jobs.append(
                {
                    "sample": sample,
                    "subsample": subsample,
                    "timestamp": timestamp,
                    "chromosome": chrom,
                    "out_tag": out_tag,
                    "regions": regions,
                }
            )
    return jobs


def _parse_out_tag_meta(out_tag: str) -> Tuple[str, str, str]:
    """Parse sample/subsample/timestamp from RESOLUTION_organism_sample_subsample_timestamp."""
    parts = out_tag.split("_")
    if len(parts) < 5:
        return out_tag, "", ""
    # drop RESOLUTION + organism
    rest = "_".join(parts[2:])
    for sample, subsample in _KNOWN_SAMPLE_SUBSAMPLE:
        prefix = f"{sample}_{subsample}_"
        if rest.startswith(prefix):
            return sample, subsample, rest[len(prefix) :]
    # fallback: first two tokens
    toks = rest.split("_")
    return toks[0], toks[1], "_".join(toks[2:])


def discover_jobs(root: str) -> List[dict]:
    """Scan flamingo_root for y outputs; metadata parsed from out_tag when possible."""
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
            tsv = os.path.join(y_dir, entry, "flamingo_coords.tsv")
            if not os.path.isfile(tsv):
                continue
            try:
                _, start_s, end_s = entry.split("_", 2)
                regions.append((int(start_s), int(end_s)))
            except ValueError:
                continue
        if not regions:
            continue
        sample, subsample, timestamp = _parse_out_tag_meta(out_tag)
        key = (out_tag, chrom)
        by_key[key] = {
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
        "--jobs_file",
        default=None,
        help="TSV: sample subsample timestamp chromosome out_tag regions(csv of start:end). "
        "If omitted, discover all y regions under --flamingo_root.",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Default: <flamingo_root>/flamingo_scc_5mb.csv",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.jobs_file:
        jobs = load_jobs(args.jobs_file)
    else:
        jobs = discover_jobs(args.flamingo_root)
        print(f"Discovered {len(jobs)} jobs under {args.flamingo_root}", flush=True)

    out_csv = args.output_csv or os.path.join(args.flamingo_root, "flamingo_scc_5mb.csv")

    rows = []
    for job in jobs:
        for start, end in job["regions"]:
            scores = scc_for_region(
                args.flamingo_root,
                job["out_tag"],
                job["chromosome"],
                start,
                end,
            )
            rows.append(
                {
                    "sample": job["sample"],
                    "subsample": job["subsample"],
                    "timestamp": job["timestamp"],
                    "chromosome": job["chromosome"],
                    "region": f"{start}-{end}",
                    **scores,
                }
            )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    n_complete = sum(
        1
        for r in rows
        if all(r.get(c) is not None for c in METHOD_COLS.values())
    )
    print(
        f"Wrote {out_csv} ({len(rows)} rows, {n_complete} with all methods)",
        flush=True,
    )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
