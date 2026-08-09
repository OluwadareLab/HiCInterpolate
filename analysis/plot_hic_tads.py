#!/usr/bin/env python3
"""Upper-triangular Hi-C heatmaps with TopDom TAD outlines (side-by-side).

Panels: Ground Truth | Ours | 4DMax | Linear | Optical Flow
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

METHODS = ("y", "pred", "4dmax", "linear", "of")
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

TAD = Tuple[int, int]


def chrom_label(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def load_topdom(bed_path: str) -> List[TAD]:
    if not os.path.isfile(bed_path) or os.path.getsize(bed_path) == 0:
        return []
    tads: List[TAD] = []
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            # chrom start end ...  OR  start end
            if len(parts) >= 3 and not parts[0].lstrip("-").isdigit():
                start, end = int(parts[1]), int(parts[2])
            else:
                start, end = int(parts[0]), int(parts[1])
            if end > start:
                tads.append((start, end))
    return tads


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


def tad_dir(tads_root: str, prefix: str, method: str) -> str:
    return os.path.join(tads_root, f"{prefix}_{method}")


def find_topdom_bed(run_dir: str) -> Optional[str]:
    path = os.path.join(run_dir, "topdom.bed")
    return path if os.path.isfile(path) else None


def print_coordinate(pos: int) -> str:
    mb = pos / 1_000_000
    if abs(mb - round(mb)) < 1e-6:
        return f"{int(round(mb))}Mb"
    return f"{mb:.1f}Mb"


def draw_triangle_panel(
    fig: plt.Figure,
    matrix: np.ndarray,
    res: int,
    start: int,
    end: int,
    tads: List[TAD],
    title: str,
    bottom: float,
    height: float,
    line_color: str = "blue",
    linewidth: float = 1.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nticks: int = 3,
):
    # Leave bottom margin for 3 Mb ticks under the triangle.
    heatmap_pos = [0.05, bottom + 0.14 * height, 0.90, 0.72 * height]

    i0 = int(round(start / res))
    i1 = int(round(end / res))
    mat = np.asarray(matrix, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    i1 = min(i1, mat.shape[0] - 1)
    i0 = max(0, i0)
    if i1 <= i0:
        raise ValueError(f"Empty region [{start}, {end}] at res={res}")
    # Square contact patch (symmetric).
    sub = mat[i0 : i1 + 1, i0 : i1 + 1]
    sub = 0.5 * (sub + sub.T)

    h_ax = fig.add_axes(heatmap_pos)
    M = sub.copy()
    mx = np.max(M)
    if mx > 0:
        M = M / mx
    n = M.shape[0]

    # Rotate square 45°: diagonal → x-axis. Keep one triangle (y >= 0).
    t = np.array([[1, 0.5], [-1, 0.5]])
    A = np.dot(
        np.array(
            [(i[1], i[0]) for i in itertools.product(range(n, -1, -1), range(0, n + 1))]
        ),
        t,
    )
    x = A[:, 1].reshape(n + 1, n + 1)
    y = A[:, 0].reshape(n + 1, n + 1)
    # Fold lower diamond half up so pcolormesh fills the kept triangle.
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
    for s, e in tads:
        if e <= start or s >= end:
            continue
        si = s // res - start // res
        ei = e // res - start // res
        si = max(0, si)
        ei = min(n - 1, ei)
        if ei - si < 2:
            continue
        px = [
            hx[:-1, :-1][n - 1 - si, si],
            hx[:-1, :-1][n - 1 - si, ei],
            hx[:-1, :-1][n - 1 - ei, ei],
        ]
        py = [
            hy[:-1, :-1][n - 1 - si, si],
            hy[:-1, :-1][n - 1 - si, ei],
            hy[:-1, :-1][n - 1 - ei, ei],
        ]
        h_ax.plot(px, py, color=line_color, linestyle="-", linewidth=linewidth)

    # Square cut along diagonal: box height/width = 1/2 → 45° sides, 90° peak.
    h_ax.set_xlim(xmin, xmax)
    h_ax.set_ylim(0.0, xmax - xmin)
    h_ax.set_box_aspect(0.5)
    h_ax.set_title(title, fontsize=24, pad=10)

    # 3 ticks on the triangle base only (start / mid / end), Mb labels.
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
    tads_root: str,
    prefix: str,
    start: int,
    end: int,
    out_path: str,
    dpi: int,
    linewidth: float,
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

    # Keep Mb tag in filename aligned with clamped window.
    parent = os.path.dirname(out_path)
    stem = os.path.basename(out_path)
    if "_topdom_" in stem:
        stem = stem.rsplit("_topdom_", 1)[0] + (
            f"_topdom_{start // 1_000_000}-{end // 1_000_000}Mb.png"
        )
        out_path = os.path.join(parent, stem)

    n_panels = len(METHODS)
    # width:height per panel = 2:1 matches half-square triangle.
    fig = plt.figure(figsize=(10.0, 5.0 * n_panels))
    panel_h = 1.0 / n_panels

    for i, method in enumerate(METHODS):
        if method not in group:
            print(f"skip missing npy: {prefix} {method}", flush=True)
            continue
        npy_path, res, chrom = group[method]
        matrix = np.load(npy_path)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix in {npy_path}, got {matrix.shape}")
        region_end = min(end, matrix.shape[0] * res)

        bed = find_topdom_bed(tad_dir(tads_root, prefix, method))
        tads = load_topdom(bed) if bed else []
        if bed is None:
            print(f"warn: no topdom.bed for {prefix}_{method}", flush=True)

        bottom = 1.0 - (i + 1) * panel_h
        draw_triangle_panel(
            fig=fig,
            matrix=matrix,
            res=res,
            start=start,
            end=region_end,
            tads=tads,
            title=METHOD_LABELS[method],
            bottom=bottom,
            height=panel_h,
            linewidth=linewidth,
        )
        print(
            f"{prefix} {method}: {len(tads)} TopDom TADs, chrom={chrom}",
            flush=True,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Side-by-side upper-triangle Hi-C + TopDom TAD outlines."
    )
    p.add_argument(
        "--npy-root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64",
    )
    p.add_argument(
        "--tads-root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/tads",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hic_tad_plots"),
    )
    p.add_argument("--start", type=int, default=55_000_000)
    p.add_argument("--end", type=int, default=60_000_000)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--linewidth", type=float, default=1.5)
    p.add_argument("--prefix", default=None, help="Single group prefix filter")
    p.add_argument("--chrom", default=None, help="Filter by chromosome number, e.g. 10")
    args = p.parse_args()

    groups = discover_npy(args.npy_root)
    if args.prefix:
        groups = {k: v for k, v in groups.items() if k == args.prefix}
    if args.chrom:
        chrom_tag = str(args.chrom).removeprefix("chr")
        groups = {
            k: v
            for k, v in groups.items()
            if any(c.endswith(chrom_tag) or c == f"chr{chrom_tag}" for _, _, c in v.values())
            or k.endswith(f"_{chrom_tag}")
        }
    if not groups:
        raise SystemExit(f"No matching .npy files under {args.npy_root}")

    for prefix in sorted(groups):
        out = os.path.join(
            args.out_dir,
            prefix,
            f"{prefix}_topdom_{args.start // 1_000_000}-{args.end // 1_000_000}Mb.png",
        )
        try:
            plot_side_by_side(
                group=groups[prefix],
                tads_root=args.tads_root,
                prefix=prefix,
                start=args.start,
                end=args.end,
                out_path=out,
                dpi=args.dpi,
                linewidth=args.linewidth,
            )
        except Exception as exc:
            print(f"error {prefix}: {exc}", flush=True)


if __name__ == "__main__":
    main()
