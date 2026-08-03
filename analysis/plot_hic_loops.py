#!/usr/bin/env python3
"""Plot upper-triangular Hi-C maps with Mustache loop circles (per method)."""

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


class Triangle:
    def __init__(
        self,
        matrix: np.ndarray,
        res: int,
        chrom: str,
        start: int,
        end: int,
        figsize=(7, 3.5),
    ):
        self.res = res
        self.chrom = chrom
        self.start = start
        self.end = end
        self.fig = plt.figure(figsize=figsize)

        i0 = int(round(start / res))
        i1 = int(round(end / res))
        mat = np.asarray(matrix, dtype=np.float64)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        i1 = min(i1, mat.shape[0] - 1)
        i0 = max(0, i0)
        if i1 <= i0:
            raise ValueError(f"Empty region [{start}, {end}] at res={res}")
        self.matrix = mat[i0 : i1 + 1, i0 : i1 + 1]
        self.cmap = JUICEBOX

    @staticmethod
    def print_coordinate(pos: int) -> str:
        i_part = int(pos) // 1_000_000
        d_part = (int(pos) % 1_000_000) // 1000
        if i_part > 0 and d_part > 0:
            return f"{i_part}M{d_part}K"
        if i_part == 0:
            return f"{d_part}K"
        return f"{i_part}M"

    def matrix_plot(
        self,
        vmin=None,
        vmax=None,
        nticks: int = 5,
        heatmap_pos=(0.1, 0.18, 0.8, 0.75),
        colorbar_pos=(0.08, 0.45, 0.02, 0.15),
        chrom_pos=(0.1, 0.08, 0.8, 0.06),
        show_cbar: bool = False,
    ):
        h_ax = self.fig.add_axes(heatmap_pos)
        M = self.matrix.copy()
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
        y[y < 0] = -y[y < 0]

        nz = M[M > 0]
        if vmax is None:
            vmax = float(np.percentile(nz, 95.99)) if nz.size else 1.0
        if vmin is None:
            vmin = float(M.min())

        sc = h_ax.pcolormesh(
            x,
            y,
            np.flipud(M),
            vmin=vmin,
            vmax=vmax,
            cmap=self.cmap,
            edgecolor="none",
            snap=True,
            linewidth=0.001,
            rasterized=True,
        )

        if show_cbar:
            c_ax = self.fig.add_axes(colorbar_pos)
            self.fig.colorbar(sc, cax=c_ax, ticks=[vmin, vmax], format="%.3g")

        xmin, xmax = A[:, 1].min(), A[:, 1].max()
        ymin = A[:, 0].min()
        h_ax.fill([xmin, xmax, xmax, xmin], [ymin, ymin, 0, 0], "w", ec="none")
        h_ax.axis("off")

        chrom_ax = self.fig.add_axes(chrom_pos)
        chrom_ax.tick_params(
            axis="both",
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelbottom=True,
            labeltop=False,
            labelleft=False,
            labelright=False,
            length=4,
            labelsize=9,
        )
        for spine in ("top", "left", "right"):
            chrom_ax.spines[spine].set_visible(False)
        chrom_ax.spines["bottom"].set_visible(True)

        ticks = list(np.linspace(0, n, nticks).astype(int))
        pos = list(np.linspace(self.start, self.end, nticks).astype(int))
        labels = [self.print_coordinate(p) for p in pos]
        chrom_ax.set_xticks(ticks)
        chrom_ax.set_xticklabels(labels)
        chrom_ax.set_xlim(0, n)
        chrom_ax.set_ylim(0, 1)
        chrom_ax.set_yticks([])

        self.heatmap_ax = h_ax
        self.chrom_ax = chrom_ax
        self.hx = x
        self.hy = y

    def _loop_pixels(
        self, loops: List[Loop]
    ) -> Tuple[np.ndarray, np.ndarray]:
        loops = [
            L
            for L in loops
            if L[0] == self.chrom
            and L[3] == self.chrom
            and L[1] >= self.start
            and L[5] < self.end
        ]
        n = self.matrix.shape[0]
        Bool = np.zeros((n, n), dtype=bool)
        for xs, xe, ys, ye in ((L[1], L[2], L[4], L[5]) for L in loops):
            s_l = range(xs // self.res - 1, int(np.ceil(xe / float(self.res))) + 1)
            e_l = range(ys // self.res - 1, int(np.ceil(ye / float(self.res))) + 1)
            si = ei = None
            for i in s_l:
                for j in e_l:
                    st = i - self.start // self.res
                    et = j - self.start // self.res
                    if 0 <= st < n and 0 <= et < n:
                        if si is None or self.matrix[st, et] > self.matrix[si, ei]:
                            si, ei = st, et
            if si is not None:
                Bool[si, ei] = True

        lx = self.hx[:-1, :-1][np.flipud(Bool)]
        ly = self.hy[:-1, :-1][np.flipud(Bool)] + 1
        return lx, ly

    def _scatter_loops(
        self,
        lx: np.ndarray,
        ly: np.ndarray,
        marker_size: float,
        marker_color: str,
        marker_type: str,
        marker_alpha: float,
    ):
        if lx.size == 0:
            return
        self.heatmap_ax.scatter(
            lx,
            ly,
            s=marker_size,
            c="none",
            marker=marker_type,
            alpha=marker_alpha,
            edgecolors=marker_color,
            linewidths=1.2,
        )

    def plot_loops(
        self,
        loops: List[Loop],
        marker_size: float = 70,
        marker_color: str = "blue",
        marker_type: str = "o",
        marker_alpha: float = 1.0,
        overlap_loops: Optional[Set[Loop]] = None,
        overlap_marker: str = "*",
        overlap_marker_color: str = "green",
        overlap_marker_size: Optional[float] = None,
    ):
        if overlap_loops is None:
            lx, ly = self._loop_pixels(loops)
            self._scatter_loops(
                lx, ly, marker_size, marker_color, marker_type, marker_alpha
            )
        else:
            unique = [L for L in loops if L not in overlap_loops]
            shared = [L for L in loops if L in overlap_loops]
            lx, ly = self._loop_pixels(unique)
            self._scatter_loops(
                lx, ly, marker_size, marker_color, marker_type, marker_alpha
            )
            ox, oy = self._loop_pixels(shared)
            osize = (
                overlap_marker_size
                if overlap_marker_size is not None
                else marker_size * 1.4
            )
            self._scatter_loops(
                ox, oy, osize, overlap_marker_color, overlap_marker, marker_alpha
            )

        self.heatmap_ax.set_xlim(self.hx.min(), self.hx.max())
        self.heatmap_ax.set_ylim(self.hy.min(), self.hy.max())

    def outfig(self, outfile: str, dpi: int = 300):
        self.fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)


def plot_one(
    npy_path: str,
    loop_tsv: Optional[str],
    res: int,
    chrom: str,
    start: int,
    end: int,
    out_path: str,
    marker_size: float,
    dpi: int,
    gt_loops: Optional[Set[Loop]] = None,
    mark_overlap: bool = False,
):
    matrix = np.load(npy_path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix in {npy_path}, got {matrix.shape}")
    if end > matrix.shape[0] * res:
        end = matrix.shape[0] * res

    loops = load_loops(loop_tsv) if loop_tsv else []
    overlap = (set(loops) & gt_loops) if (mark_overlap and gt_loops) else None
    tri = Triangle(matrix, res=res, chrom=chrom, start=start, end=end)
    tri.matrix_plot()
    if loops:
        tri.plot_loops(
            loops,
            marker_size=marker_size,
            marker_color="blue",
            overlap_loops=overlap,
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tri.outfig(out_path, dpi=dpi)
    n_ov = len(overlap) if overlap is not None else 0
    print(
        f"wrote {out_path} ({len(loops)} loops, {n_ov} overlap GT)",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(
        description="Upper-triangle Hi-C + Mustache loop circles for each method."
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
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
    )
    p.add_argument("--prefix", default=None, help="Optional single group prefix filter")
    args = p.parse_args()

    groups = discover_npy(args.npy_root)
    if args.prefix:
        groups = {k: v for k, v in groups.items() if k == args.prefix}
    if not groups:
        raise SystemExit(f"No matching .npy files under {args.npy_root}")

    for prefix in sorted(groups):
        gt_tsv = find_mustache_tsv(mustache_dir(args.mustache_root, prefix, "y"))
        gt_loops = load_loop_set(gt_tsv)
        for method in args.methods:
            if method not in groups[prefix]:
                print(f"skip missing npy: {prefix} {method}", flush=True)
                continue
            npy_path, res, chrom = groups[prefix][method]
            mdir = mustache_dir(args.mustache_root, prefix, method)
            tsv = find_mustache_tsv(mdir)
            if tsv is None:
                print(f"warn: no mustache tsv for {mdir}", flush=True)
            label = METHOD_LABELS[method].replace(" ", "_")
            out = os.path.join(
                args.out_dir,
                prefix,
                f"{prefix}_{method}_{label}_{args.start // 1_000_000}-{args.end // 1_000_000}Mb.png",
            )
            try:
                plot_one(
                    npy_path=npy_path,
                    loop_tsv=tsv,
                    res=res,
                    chrom=chrom,
                    start=args.start,
                    end=args.end,
                    out_path=out,
                    marker_size=args.marker_size,
                    dpi=args.dpi,
                    gt_loops=gt_loops,
                    mark_overlap=method in COMPARE_METHODS,
                )
            except Exception as exc:
                print(f"error {prefix} {method}: {exc}", flush=True)


if __name__ == "__main__":
    main()
