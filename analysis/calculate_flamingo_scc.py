#!/usr/bin/env python3
"""SCC between FLAMINGO structures: each method vs y, per genomic region."""

import argparse
import csv
import os
import sys

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def load_coords_tsv(path: str):
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
    # Squared distances then sqrt; upper triangle only for SCC
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


def scc_aligned(frag_a, xyz_a, frag_b, xyz_b) -> tuple:
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
    if spearmanr is not None:
        corr, _ = spearmanr(flat_a, flat_b)
        corr = float(corr)
    else:
        corr = spearman_manual(flat_a, flat_b)
    return corr, int(common.size)


def parse_region(s: str):
    s = s.strip()
    if ":" in s:
        a, b = s.split(":", 1)
    elif "-" in s:
        a, b = s.split("-", 1)
    else:
        raise argparse.ArgumentTypeError(f"region must be START:END, got {s!r}")
    start, end = int(a), int(b)
    if end <= start:
        raise argparse.ArgumentTypeError(f"region end must be > start: {s!r}")
    return start, end


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flamingo_root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/flamingo_user",
    )
    p.add_argument("--chrom", type=int, default=10)
    p.add_argument("--sample_tag", default="25000_human_dmso_control_dmso_control_60m")
    p.add_argument("--methods", nargs="+", default=["pred", "of", "linear", "4dmax"])
    p.add_argument(
        "--region",
        action="append",
        type=parse_region,
        default=None,
        help="Bin region START:END (repeatable). Default: 4 equal splits of --n_bins",
    )
    p.add_argument("--n_bins", type=int, default=5352)
    p.add_argument("--n_regions", type=int, default=4)
    p.add_argument(
        "--output_csv",
        default=None,
        help="Default: <flamingo_root>/<sample_tag>_<chrom>_flamingo_scc.csv",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.region:
        regions = args.region
    else:
        region_len = args.n_bins // args.n_regions
        regions = []
        for r in range(args.n_regions):
            start = r * region_len
            end = args.n_bins if r == args.n_regions - 1 else (r + 1) * region_len
            regions.append((start, end))

    out_csv = args.output_csv or os.path.join(
        args.flamingo_root,
        f"{args.sample_tag}_{args.chrom}_flamingo_scc.csv",
    )

    rows = []
    for start, end in regions:
        y_dir = os.path.join(
            args.flamingo_root,
            f"{args.sample_tag}_{args.chrom}_y",
            f"region_{start}_{end}",
        )
        y_tsv = os.path.join(y_dir, "flamingo_coords.tsv")
        if not os.path.isfile(y_tsv):
            print(f"SKIP missing y coords: {y_tsv}", flush=True)
            continue
        frag_y, xyz_y = load_coords_tsv(y_tsv)

        for method in args.methods:
            m_dir = os.path.join(
                args.flamingo_root,
                f"{args.sample_tag}_{args.chrom}_{method}",
                f"region_{start}_{end}",
            )
            m_tsv = os.path.join(m_dir, "flamingo_coords.tsv")
            if not os.path.isfile(m_tsv):
                print(f"SKIP missing {method} coords: {m_tsv}", flush=True)
                continue
            frag_m, xyz_m = load_coords_tsv(m_tsv)
            scc, n_common = scc_aligned(frag_y, xyz_y, frag_m, xyz_m)
            print(
                f"chr{args.chrom} region {start}-{end} {method} vs y: "
                f"SCC={scc:.6f} n={n_common}",
                flush=True,
            )
            rows.append(
                {
                    "chromosome": args.chrom,
                    "region": f"{start}-{end}",
                    "method": method,
                    "n_common": n_common,
                    "scc": scc,
                }
            )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chromosome", "region", "method", "n_common", "scc"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
