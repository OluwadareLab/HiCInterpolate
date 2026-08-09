#!/usr/bin/env python3
"""Upper-triangular Hi-C heatmaps with Mustache loop circles (top-to-bottom).

Panels: Ground Truth | Ours | 4DMax | Linear | Optical Flow
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

Loop = Tuple[str, int, int, str, int, int]

METHODS = ("y", "pred", "4dmax", "linear", "of")
COMPARE_METHODS = ("pred", "4dmax", "linear", "of")
METHOD_LABELS = {
    "y": "Ground Truth",
    "pred": "Ours",
    "4dmax": "4DMax",
    "linear": "Linear",
    "of": "Optical Flow",
}

NPY_RE = re.compile(
    r"^(?P<res>\d+)_(?P<window>\d+)_(?P<body>.+)_(?P<method>y|pred|linear|of|4dmax)\.npy$"
)

JUICEBOX = LinearSegmentedColormap.from_list(
    "juicebox", ["#FFFFFF", "#FFDFDF", "#FF7575", "#FF2626", "#F70000"], N=256
)


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def print_coordinate(pos: int) -> str:
    mb = pos / 1_000_000
    if abs(mb - round(mb)) < 1e-6:
        return f"{int(round(mb))}Mb"
    return f"{mb:.1f}Mb"


def load_loops(tsv_path: str) -> List[Loop]:
    if not os.path.isfile(tsv_path) or os.path.getsize(tsv_path) == 0:
        return []
    loops = []
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            loops.append(
                (
                    chrom_label(row["BIN1_CHR"]),
                    int(row["BIN1_START"]),
                    int(row["BIN1_END"]),
                    chrom_label(row["BIN2_CHROMOSOME"]),
                    int(row["BIN2_START"]),
                    int(row["BIN2_END"]),
                )
            )
    return loops


def load_loop_set(tsv_path: Optional[str]) -> Set[Loop]:
    if not tsv_path:
        return set()
    return set(load_loops(tsv_path))


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


def discover_npy(
    npy_root: str,
) -> Dict[str, Dict[str, Tuple[str, int, str]]]:
    """prefix -> method -> (npy_path, resolution, chrom)."""
    groups: Dict[str, Dict[str, Tuple[str, int, str]]] = {}
    for name in sorted(os.listdir(npy_root)):
        m = NPY_RE.match(name)
        if not m:
            continue
        res = int(m.group("res"))
        body = m.group("body")
        method = m.group("method")
        chrom = body.rsplit("_", 1)[-1]
        prefix = f"{res}_{body}"
        groups.setdefault(prefix, {})[method] = (
            os.path.join(npy_root, name),
            res,
            chrom_label(chrom),
        )
    return groups


def mustache_dir(mustache_root: str, prefix: str, method: str) -> str:
    return os.path.join(mustache_root, f"{prefix}_{method}")


def loop_pixels(
    matrix: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    res: int,
    chrom: str,
    start: int,
    end: int,
    loops: List[Loop],
) -> Tuple[np.ndarray, np.ndarray]:
    loops = [
        L
        for L in loops
        if L[0] == chrom
        and L[3] == chrom
        and L[1] >= start
        and L[5] < end
    ]
    n = matrix.shape[0]
    Bool = np.zeros((n, n), dtype=bool)
    for xs, xe, ys, ye in ((L[1], L[2], L[4], L[5]) for L in loops):
        s_l = range(xs // res - 1, int(np.ceil(xe / float(res))) + 1)
        e_l = range(ys // res - 1, int(np.ceil(ye / float(res))) + 1)
        si = ei = None
        for i in s_l:
            for j in e_l:
                st = i - start // res
                et = j - start // res
                if 0 <= st < n and 0 <= et < n:
                    if si is None or matrix[st, et] > matrix[si, ei]:
                        si, ei = st, et
        if si is not None:
            Bool[si, ei] = True

    lx = hx[:-1, :-1][np.flipud(Bool)]
    ly = hy[:-1, :-1][np.flipud(Bool)] + 1
    return lx, ly


def scatter_loops(
    ax,
    lx: np.ndarray,
    ly: np.ndarray,
    marker_size: float,
    marker_color: str,
    marker_type: str,
    marker_alpha: float,
):
    if lx.size == 0:
        return
    ax.scatter(
        lx,
        ly,
        s=marker_size,
        c="none",
        marker=marker_type,
        alpha=marker_alpha,
        edgecolors=marker_color,
        linewidths=1.2,
    )


def draw_triangle_panel(
    fig: plt.Figure,
    matrix: np.ndarray,
    res: int,
    chrom: str,
    start: int,
    end: int,
    loops: List[Loop],
    title: str,
    bottom: float,
    height: float,
    marker_size: float = 30,
    overlap_loops: Optional[Set[Loop]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nticks: int = 3,
):
    heatmap_pos = [0.05, bottom + 0.14 * height, 0.90, 0.72 * height]

    i0 = int(round(start / res))
    i1 = int(round(end / res))
    mat = np.asarray(matrix, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    i1 = min(i1, mat.shape[0] - 1)
    i0 = max(0, i0)
    if i1 <= i0:
        raise ValueError(f"Empty region [{start}, {end}] at res={res}")
    sub = mat[i0 : i1 + 1, i0 : i1 + 1]
    sub = 0.5 * (sub + sub.T)

    h_ax = fig.add_axes(heatmap_pos)
    M = sub.copy()
    mx = np.max(M)
    if mx > 0:
        M = M / mx
    n = M.shape[0]

    t = np.array([[1, 0.5], [-1, 0.5]])
    A = np.dot(
        np.array(
            [(i[1], i[0]) for i in itertools.product(range(n, -1, -1), range(0, n + 1))]
        ),
        t,
    )
    x = A[:, 1].reshape(n + 1, n + 1)
    y = A[:, 0].reshape(n + 1, n + 1)
    y_plot = np.abs(y)

    nz = M[M > 0]
    if vmax is None:
        vmax = float(np.percentile(nz, 95.99)) if nz.size else 1.0
    if vmin is None:
        vmin = float(M.min())

    h_ax.pcolormesh(
        x,
        y_plot,
        np.flipud(M),
        vmin=vmin,
        vmax=vmax,
        cmap=JUICEBOX,
        edgecolor="none",
        snap=True,
        linewidth=0.001,
        rasterized=True,
    )
    xmin, xmax = float(x.min()), float(x.max())
    h_ax.fill(
        [xmin, xmax, xmax, xmin],
        [float(y.min()), float(y.min()), 0, 0],
        "w",
        ec="none",
    )

    hx, hy = x, y_plot
    if loops:
        if overlap_loops is None:
            lx, ly = loop_pixels(sub, hx, hy, res, chrom, start, end, loops)
            scatter_loops(h_ax, lx, ly, marker_size, "blue", "o", 1.0)
        else:
            unique = [L for L in loops if L not in overlap_loops]
            shared = [L for L in loops if L in overlap_loops]
            lx, ly = loop_pixels(sub, hx, hy, res, chrom, start, end, unique)
            scatter_loops(h_ax, lx, ly, marker_size, "blue", "o", 1.0)
            ox, oy = loop_pixels(sub, hx, hy, res, chrom, start, end, shared)
            scatter_loops(h_ax, ox, oy, marker_size * 4, "green", "*", 1.0)

    h_ax.set_xlim(xmin, xmax)
    h_ax.set_ylim(0.0, xmax - xmin)
    h_ax.set_box_aspect(0.5)
    h_ax.set_title(title, fontsize=24, pad=10)

    tick_x = np.linspace(xmin, xmax, nticks)
    tick_bp = np.linspace(start, end, nticks)
    h_ax.set_xticks(tick_x)
    h_ax.set_xticklabels([print_coordinate(int(p)) for p in tick_bp])
    h_ax.set_yticks([])
    h_ax.minorticks_off()
    for spine in ("top", "left", "right"):
        h_ax.spines[spine].set_visible(False)
    h_ax.spines["bottom"].set_visible(True)
    h_ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        length=5,
        labelsize=18,
        pad=3,
    )
    h_ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def plot_side_by_side(
    group: Dict[str, Tuple[str, int, str]],
    mustache_root: str,
    prefix: str,
    start: int,
    end: int,
    out_path: str,
    dpi: int,
    marker_size: float,
):
    # Clamp window into chromosome length (e.g. chr21 < 55Mb).
    chrom_len = None
    for method in METHODS:
        if method in group:
            npy_path, res, _ = group[method]
            matrix = np.load(npy_path, mmap_mode="r")
            chrom_len = int(matrix.shape[0]) * res
            break
    if chrom_len is None:
        raise ValueError(f"No matrices for {prefix}")
    win = end - start
    if start >= chrom_len:
        start = max(0, chrom_len - win)
        end = chrom_len
        print(f"clamp region -> [{start}, {end}] for {prefix}", flush=True)
    elif end > chrom_len:
        end = chrom_len
        if end - start < win // 5:
            start = max(0, end - win)
        print(f"clamp end -> [{start}, {end}] for {prefix}", flush=True)

    parent = os.path.dirname(out_path)
    stem = os.path.basename(out_path)
    if "_mustache_" in stem:
        stem = stem.rsplit("_mustache_", 1)[0] + (
            f"_mustache_{start // 1_000_000}-{end // 1_000_000}Mb.png"
        )
        out_path = os.path.join(parent, stem)

    n_panels = len(METHODS)
    fig = plt.figure(figsize=(10.0, 5.0 * n_panels))
    panel_h = 1.0 / n_panels

    gt_tsv = find_mustache_tsv(mustache_dir(mustache_root, prefix, "y"))
    gt_loops = load_loop_set(gt_tsv)

    for i, method in enumerate(METHODS):
        if method not in group:
            print(f"skip missing npy: {prefix} {method}", flush=True)
            continue
        npy_path, res, chrom = group[method]
        matrix = np.load(npy_path)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix in {npy_path}, got {matrix.shape}")
        region_end = min(end, matrix.shape[0] * res)

        tsv = find_mustache_tsv(mustache_dir(mustache_root, prefix, method))
        loops = load_loops(tsv) if tsv else []
        if tsv is None:
            print(f"warn: no mustache tsv for {prefix}_{method}", flush=True)

        overlap = (
            (set(loops) & gt_loops) if method in COMPARE_METHODS and gt_loops else None
        )
        bottom = 1.0 - (i + 1) * panel_h
        draw_triangle_panel(
            fig=fig,
            matrix=matrix,
            res=res,
            chrom=chrom,
            start=start,
            end=region_end,
            loops=loops,
            title=METHOD_LABELS[method],
            bottom=bottom,
            height=panel_h,
            marker_size=marker_size,
            overlap_loops=overlap,
        )
        n_ov = len(overlap) if overlap is not None else 0
        print(
            f"{prefix} {method}: {len(loops)} loops, {n_ov} overlap GT, chrom={chrom}",
            flush=True,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Side-by-side upper-triangle Hi-C + Mustache loop circles."
    )
    p.add_argument(
        "--npy-root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64",
    )
    p.add_argument(
        "--mustache-root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/mustache",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hic_loop_plots"),
    )
    p.add_argument("--start", type=int, default=55_000_000)
    p.add_argument("--end", type=int, default=60_000_000)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--marker-size", type=float, default=30)
    p.add_argument("--prefix", default=None, help="Optional single group prefix filter")
    p.add_argument("--chrom", default=None, help="Filter by chromosome number, e.g. 10")
    args = p.parse_args()

    groups = discover_npy(args.npy_root)
    if args.prefix:
        groups = {k: v for k, v in groups.items() if k == args.prefix}
    if args.chrom:
        chrom_tag = str(args.chrom).removeprefix("chr")
        groups = {k: v for k, v in groups.items() if k.endswith(f"_{chrom_tag}")}
    if not groups:
        raise SystemExit(f"No matching .npy files under {args.npy_root}")

    for prefix in sorted(groups):
        out = os.path.join(
            args.out_dir,
            prefix,
            f"{prefix}_mustache_{args.start // 1_000_000}-{args.end // 1_000_000}Mb.png",
        )
        try:
            plot_side_by_side(
                group=groups[prefix],
                mustache_root=args.mustache_root,
                prefix=prefix,
                start=args.start,
                end=args.end,
                out_path=out,
                dpi=args.dpi,
                marker_size=args.marker_size,
            )
        except Exception as exc:
            print(f"error {prefix}: {exc}", flush=True)


if __name__ == "__main__":
    main()
