from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import zoom
from scipy.signal import find_peaks

TAD = Tuple[int, int]


CB_BLACK = "#000000"
CB_ORANGE = "#E69F00"
CB_SKY = "#56B4E9"
CB_GREEN = "#009E73"
CB_YELLOW = "#F0E442"
CB_BLUE = "#0072B2"
CB_VERMILLION = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_GREY = "#999999"

# Juicebox-style Hi-C sequential (white → red)
JUICEBOX = LinearSegmentedColormap.from_list(
    "juicebox",
    ["#FFFFFF", "#FFDFDF", "#FF7575", "#FF2626", "#F70000"],
    N=256,
)
# Diverging differential: blue ↔ white ↔ vermillion (no red–green)
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "cb_diff",
    [CB_BLUE, CB_SKY, "#F7F7F7", CB_ORANGE, CB_VERMILLION],
    N=256,
)

STAGE_LABELS = {
    "early2_cell": "Early 2-cell",
    "late2_cell": "Late 2-cell",
    "8cell": "8-cell",
}
STAGE_COLORS = {
    "early2_cell": CB_GREEN,
    "late2_cell": CB_ORANGE,
    "8cell": CB_BLUE,
}

# Journal fonts (readable when placed at ~1-col width of a 2-col template)
FONT_FAMILY = "Arial"
FS_PANEL = 11     # A/B/C
FS_LABEL = 10     # axis labels
FS_TICK = 8       # ticks / colorbar
FS_TITLE = 10     # stage titles, Δ formulas
FS_LEGEND = 9
FW_PANEL = "normal"
FW_LABEL = "normal"

NPY_RE = re.compile(
    r"^(?P<res>\d+)_(?P<window>\d+)_(?P<body>.+)_(?P<method>y|pred|linear|of|4dmax)\.npy$"
)


def _apply_journal_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Liberation Sans", "DejaVu Sans", "Noto Sans"],
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_TITLE,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,  # TrueType — journal editable text
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "axes.unicode_minus": False,
        }
    )


def find_npy(
    npy_root: str,
    resolution: int,
    stage_tag: str,
    chrom: str,
    method: str,
) -> Optional[str]:
    chrom_tag = str(chrom).removeprefix("chr")
    suffix = f"_{chrom_tag}_{method}.npy"
    for name in os.listdir(npy_root):
        if not name.endswith(suffix):
            continue
        m = NPY_RE.match(name)
        if not m or int(m.group("res")) != resolution:
            continue
        body = m.group("body")
        if not body.endswith(f"_{chrom_tag}"):
            continue
        body_no_chrom = body[: -(len(chrom_tag) + 1)]
        if body_no_chrom == stage_tag or body_no_chrom.endswith(stage_tag):
            return os.path.join(npy_root, name)
    return None


def load_matrix(path: str) -> np.ndarray:
    mat = np.asarray(np.load(path), dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (mat + mat.T)


def insulation_score_raw(matrix: np.ndarray, window: int) -> np.ndarray:
    """Crane-style raw IS: mean contacts in the square across the diagonal."""
    n = matrix.shape[0]
    scores = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n - window):
        scores[i] = float(np.mean(matrix[i - window : i, i + 1 : i + 1 + window]))
    return scores


def insulation_score(matrix: np.ndarray, window: int) -> np.ndarray:
    """log2(IS / chromosomal mean) — Nature 2017 Fig. 2b convention.

    Flanks sit near 0; strong TAD boundaries appear as deep negative dips
    (approx. −0.5 to −1), not as raw contact counts.
    """
    raw = insulation_score_raw(matrix, window)
    pos = raw[np.isfinite(raw) & (raw > 0)]
    if pos.size == 0:
        return raw
    mu = float(np.mean(pos))
    out = np.full_like(raw, np.nan)
    m = np.isfinite(raw) & (raw > 0)
    out[m] = np.log2(raw[m] / mu)
    return out


def call_tads_from_insulation(
    scores: np.ndarray,
    res: int,
    min_prominence: float = 0.1,
    min_distance_bins: int = 8,
    min_tad_bins: int = 4,
    max_tad_bins: int = 200,
    bin_start: int = 0,
    bin_end: Optional[int] = None,
) -> Tuple[List[TAD], List[int], np.ndarray]:
    """TADs = intervals between IS local minima (boundaries).

    `scores` should be log2-normalized IS (low / negative at boundaries).
    """
    n = scores.shape[0]
    b0 = max(0, bin_start)
    b1 = n if bin_end is None else min(n, bin_end)
    if b1 - b0 < 2 * min_tad_bins:
        return [], [], scores

    # Peaks on inverted track = insulation dips.
    track = np.where(np.isfinite(scores), -scores, -np.inf)
    track[:b0] = -np.inf
    track[b1:] = -np.inf

    peaks, _ = find_peaks(
        track, distance=min_distance_bins, prominence=min_prominence
    )
    boundary_bins = sorted(int(p) for p in peaks)
    if not boundary_bins:
        return [], [], scores

    edges = sorted(set([b0] + boundary_bins + [b1]))
    tads: List[TAD] = []
    for i in range(len(edges) - 1):
        s_bin, e_bin = edges[i], edges[i + 1]
        length = e_bin - s_bin
        if min_tad_bins <= length <= max_tad_bins:
            tads.append((s_bin * res, e_bin * res))
    return tads, boundary_bins, scores


def filter_tads_by_bins(
    tads: Sequence[TAD], res: int, bin_start: int, bin_end: int
) -> List[TAD]:
    start_bp = bin_start * res
    end_bp = bin_end * res
    return [(s, e) for s, e in tads if e > start_bp and s < end_bp]


def extract_tad_window(
    matrix: np.ndarray, start_bp: int, end_bp: int, res: int, pad_frac: float = 0.5
) -> Optional[np.ndarray]:
    """Submatrix for TAD ± pad_frac * TAD length (Fig. 2a)."""
    length = end_bp - start_bp
    if length <= 0:
        return None
    pad = int(round(pad_frac * length))
    i0 = max(0, (start_bp - pad) // res)
    i1 = min(matrix.shape[0], int(np.ceil((end_bp + pad) / res)))
    if i1 - i0 < 4:
        return None
    return matrix[i0:i1, i0:i1]


def resize_square(mat: np.ndarray, size: int) -> np.ndarray:
    if mat.shape[0] == size and mat.shape[1] == size:
        return mat.astype(np.float64, copy=True)
    return zoom(mat, (size / mat.shape[0], size / mat.shape[1]), order=1)


def aggregate_tads(
    matrix: np.ndarray,
    tads: Sequence[TAD],
    res: int,
    grid: int,
    min_bins: int = 4,
) -> Tuple[np.ndarray, int]:
    acc = np.zeros((grid, grid), dtype=np.float64)
    count = 0
    for start, end in tads:
        if (end - start) // res < min_bins:
            continue
        sub = extract_tad_window(matrix, start, end, res)
        if sub is None:
            continue
        acc += resize_square(sub, grid)
        count += 1
    if count == 0:
        return acc, 0
    mean_mat = acc / count
    mu = float(np.mean(mean_mat))
    if mu > 0:
        mean_mat /= mu
    return mean_mat, count


def profile_insulation_around_tads(
    scores: np.ndarray,
    tads: Sequence[TAD],
    res: int,
    n_points: int = 101,
    pad_frac: float = 0.5,
) -> np.ndarray:
    """Mean IS across each ref TAD ± pad_frac·L, scaled to a common axis.

    Matches Du et al. Fig. 2b / 2a window: total span = 2× TAD length.
    x=0 is the TAD *center* (high IS inside domain); boundaries sit near
    ±0.25 of the axis (dips). Not centered on the boundary.
    """
    n = scores.shape[0]
    # Relative coordinate: 0 = TAD center; ±(0.5+pad_frac) = outer edges.
    half_span = 0.5 + pad_frac  # 1.0 when pad_frac=0.5 → "2 × TAD"
    x_rel = np.linspace(-half_span, half_span, n_points)
    profiles: List[np.ndarray] = []

    for start_bp, end_bp in tads:
        length_bp = end_bp - start_bp
        if length_bp <= 0:
            continue
        length_bins = max(length_bp / res, 1.0)
        center_bin = 0.5 * (start_bp + end_bp) / res
        half_bins = int(np.ceil(half_span * length_bins))
        i0 = int(np.floor(center_bin - half_bins))
        i1 = int(np.ceil(center_bin + half_bins))
        if i0 < 0 or i1 >= n:
            continue
        seg = scores[i0 : i1 + 1]
        if seg.size < 5 or np.any(~np.isfinite(seg)):
            continue
        x_bins = (np.arange(i0, i1 + 1) - center_bin) / length_bins
        profiles.append(np.interp(x_rel, x_bins, seg))

    if not profiles:
        return np.full(n_points, np.nan)
    return np.nanmean(np.vstack(profiles), axis=0)


def random_tads(
    start_bp: int,
    end_bp: int,
    lengths: Sequence[int],
    rng: np.random.Generator,
) -> List[TAD]:
    """Shuffle TAD positions (keep lengths) as Fig. 2b random control."""
    span = end_bp - start_bp
    out: List[TAD] = []
    for L in lengths:
        if L <= 0 or L >= span:
            continue
        s = int(rng.integers(start_bp, end_bp - L))
        out.append((s, s + L))
    return out


def plot_figure2(
    stage_aggs: Dict[str, np.ndarray],
    stages: Sequence[str],
    ins_curves: Dict[str, np.ndarray],
    x_rel: np.ndarray,
    out_path: str,
    dpi: int,
):
    """Combined Fig. 2 layout (restored spacing) + Juicebox TAD heatmaps."""
    _apply_journal_style()
    n = len(stages)
    n_diff = n - 1

    # Restored non-overlapping layout (inches)
    panel = 1.85
    cbar_w = 0.10
    cbar_pad = 0.05
    gap_agg = 0.45
    gap_diff = 0.75
    row_gap_agg_diff = 0.12
    row_gap_diff_ins = 0.32
    left_m = 1.05
    right_m = 0.35
    top_m = 0.14
    bottom_m = 0.35
    ins_h = 1.45
    diff_title_h = 0.14
    panel_label_h = 0.0
    label_x = 0.06

    cell_agg = panel + cbar_pad + cbar_w
    cell_diff = panel + cbar_pad + cbar_w
    top_w = n * cell_agg + (n - 1) * gap_agg
    mid_w = n_diff * cell_diff + max(n_diff - 1, 0) * gap_diff
    fig_w = left_m + top_w + right_m
    fig_h = (
        top_m
        + panel_label_h
        + panel
        + row_gap_agg_diff
        + diff_title_h
        + panel
        + row_gap_diff_ins
        + panel_label_h
        + ins_h
        + bottom_m
    )

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    def ax_box(x_in: float, y_in: float, w_in: float, h_in: float):
        return fig.add_axes(
            [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]
        )

    vmax = max(
        float(np.percentile(m[m > 0], 99)) if np.any(m > 0) else 1.0
        for m in stage_aggs.values()
    )
    vmax = max(vmax, 1.0)

    y_ins = bottom_m
    y_ins_label = y_ins + ins_h
    y_diff = y_ins_label + panel_label_h + row_gap_diff_ins
    y_diff_title = y_diff + panel
    y_agg = y_diff_title + diff_title_h + row_gap_agg_diff

    # --- Aggregate TADs ---
    for j, stage in enumerate(stages):
        x0 = left_m + j * (cell_agg + gap_agg)
        ax = ax_box(x0, y_agg, panel, panel)
        im = ax.imshow(
            stage_aggs[stage],
            cmap=JUICEBOX,
            vmin=0,
            vmax=vmax,
            origin="upper",
            aspect="auto",
        )
        g = stage_aggs[stage].shape[0]
        lo, hi = g * 0.25, g * 0.75
        ax.plot(
            [lo, hi, hi, lo, lo],
            [lo, lo, hi, hi, lo],
            color="#0000FF",
            lw=1.0,
        )
        ax.set_title(
            STAGE_LABELS.get(stage, stage),
            fontsize=FS_TITLE,
            fontweight=FW_LABEL,
            pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(
                "8-cell TADs", fontsize=FS_LABEL, fontweight=FW_LABEL, labelpad=4
            )
        cax = ax_box(x0 + panel + cbar_pad, y_agg, cbar_w, panel)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)

    # --- Differentials ---
    x_mid0 = left_m + 0.5 * (top_w - mid_w)
    for j in range(n_diff):
        a, b = stages[j], stages[j + 1]
        diff = stage_aggs[b] - stage_aggs[a]
        x0 = x_mid0 + j * (cell_diff + gap_diff)
        fig.text(
            (x0 + 0.5 * panel) / fig_w,
            (y_diff_title + 0.02) / fig_h,
            f"{STAGE_LABELS.get(b, b)} - {STAGE_LABELS.get(a, a)}",
            fontsize=FS_TITLE,
            fontweight=FW_LABEL,
            ha="center",
            va="bottom",
        )
        ax = ax_box(x0, y_diff, panel, panel)
        lim = float(np.percentile(np.abs(diff), 98)) or 1.0
        im = ax.imshow(
            diff,
            cmap=DIFF_CMAP,
            norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
            origin="upper",
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(
                "Differential", fontsize=FS_LABEL, fontweight=FW_LABEL, labelpad=4
            )
        cax = ax_box(x0 + panel + cbar_pad, y_diff, cbar_w, panel)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)

    # --- Insulation ---
    ax = ax_box(left_m, y_ins, top_w, ins_h)
    handles = []
    labels = []
    for stage in stages:
        (ln,) = ax.plot(
            x_rel,
            ins_curves[stage],
            color=STAGE_COLORS.get(stage, CB_BLACK),
            lw=1.6,
        )
        handles.append(ln)
        labels.append(STAGE_LABELS.get(stage, stage))

    ax.axvline(0.0, color=CB_BLACK, lw=0.7, ls=":")
    ax.axhline(0.0, color=CB_GREY, lw=0.5, ls=":", alpha=0.8)
    ax.axvline(-0.5, color=CB_GREY, lw=0.6, ls="--", alpha=0.7)
    ax.axvline(0.5, color=CB_GREY, lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel(
        "Insulation score  log2(IS / mean)",
        fontsize=FS_LABEL,
        fontweight=FW_LABEL,
        labelpad=6,
    )
    ax.tick_params(labelsize=FS_TICK, width=0.5, length=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["2xTAD", "boundary", "TAD", "boundary", "2xTAD"])

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(stages),
        frameon=False,
        fontsize=FS_LEGEND,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stem, _ = os.path.splitext(out_path)
    png_path = stem + ".png"
    pdf_path = stem + ".pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote {png_path}", flush=True)
    print(f"wrote {pdf_path}", flush=True)



def save_tads_bed(path: str, tads: Sequence[TAD], chrom: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for s, e in tads:
            fh.write(f"{chrom}\t{s}\t{e}\n")


def run_chrom(
    chrom: str,
    stages: Sequence[str],
    ref_stage: str,
    stage_prefix: str,
    npy_root: str,
    resolution: int,
    method: str,
    grid: int,
    ins_window: int,
    flank_tad_frac: float,
    n_profile: int,
    out_dir: str,
    dpi: int,
    seed: int,
    bin_start: Optional[int],
    bin_end: Optional[int],
    min_prominence: float,
    min_distance_bins: int,
    min_tad_bins: int,
    max_tad_bins: int,
):
    chrom_tag = str(chrom).removeprefix("chr")
    chrom_name = f"chr{chrom_tag}"
    if ref_stage not in stages:
        stages = list(stages) + [ref_stage]

    matrices: Dict[str, np.ndarray] = {}
    for stage in stages:
        tag = f"{stage_prefix}_{stage}"
        path = find_npy(npy_root, resolution, tag, chrom_tag, method)
        if path is None:
            raise SystemExit(f"Missing npy for {tag} chr{chrom_tag} method={method}")
        matrices[stage] = load_matrix(path)
        print(
            f"{chrom_name} {stage}: {path} shape={matrices[stage].shape}", flush=True
        )

    n_bins = matrices[ref_stage].shape[0]
    b0 = 0 if bin_start is None else int(bin_start)
    b1 = n_bins if bin_end is None else int(bin_end)
    if b0 >= n_bins:
        print(
            f"skip {chrom_name}: bin region [{b0},{b1}) beyond {n_bins} bins",
            flush=True,
        )
        return
    b1 = min(b1, n_bins)
    if b1 - b0 < 4:
        print(f"skip {chrom_name}: bin region [{b0},{b1}) too small", flush=True)
        return
    region_tag = "full" if bin_start is None and bin_end is None else f"bins{b0}-{b1}"

    # --- Fig. 2 reference: call TADs ONLY on 8-cell (ref_stage) ---
    ref_scores = insulation_score(matrices[ref_stage], ins_window)
    tads, boundary_bins, rel_scores = call_tads_from_insulation(
        ref_scores,
        res=resolution,
        min_prominence=min_prominence,
        min_distance_bins=min_distance_bins,
        min_tad_bins=min_tad_bins,
        max_tad_bins=max_tad_bins,
        bin_start=b0,
        bin_end=b1,
    )
    tads = filter_tads_by_bins(tads, resolution, b0, b1)
    if not tads:
        print(f"skip {chrom_name}: no IS TADs from {ref_stage} in {region_tag}", flush=True)
        return

    print(
        f"{chrom_name}: {len(tads)} TADs / {len(boundary_bins)} boundaries "
        f"called on {ref_stage}; reused for all stages ({region_tag})",
        flush=True,
    )

    chrom_out = os.path.join(out_dir, chrom_name)
    os.makedirs(chrom_out, exist_ok=True)
    stem = f"{chrom_name}_{method}_{region_tag}"
    save_tads_bed(os.path.join(chrom_out, f"{stem}_ref8cell_is_tads.bed"), tads, chrom_name)
    np.savez_compressed(
        os.path.join(chrom_out, f"{stem}_ref8cell_is_scores.npz"),
        scores=ref_scores,
        rel_scores=rel_scores,
        boundary_bins=np.asarray(boundary_bins, dtype=np.int64),
        bin_start=b0,
        bin_end=b1,
        ref_stage=ref_stage,
    )

    # --- Fig. 2a: aggregate + differential using fixed 8-cell TADs ---
    agg: Dict[str, np.ndarray] = {}
    for stage in stages:
        mat, n_used = aggregate_tads(matrices[stage], tads, resolution, grid)
        agg[stage] = mat
        print(f"{chrom_name} {stage}: aggregated {n_used} ref-{ref_stage} TADs", flush=True)

    # --- Fig. 2b: IS over the same 8-cell TAD windows for every stage ---
    # Centered on TAD body (high at 0), boundaries near ±0.5 — not on boundary.
    half_span = 0.5 + flank_tad_frac
    x_rel = np.linspace(-half_span, half_span, n_profile)
    curves: Dict[str, np.ndarray] = {}
    for stage in stages:
        scores = insulation_score(matrices[stage], ins_window)
        curves[stage] = profile_insulation_around_tads(
            scores,
            tads,
            resolution,
            n_points=n_profile,
            pad_frac=flank_tad_frac,
        )
        print(
            f"{chrom_name} {stage}: IS profile centered on {ref_stage} TADs",
            flush=True,
        )

    plot_figure2(
        stage_aggs=agg,
        stages=stages,
        ins_curves=curves,
        x_rel=x_rel,
        out_path=os.path.join(chrom_out, f"{stem}_figure2.png"),
        dpi=dpi,
    )
    np.savez_compressed(
        os.path.join(chrom_out, f"{stem}_figure2.npz"),
        x_rel=x_rel,
        bin_start=b0,
        bin_end=b1,
        ref_stage=np.asarray(ref_stage),
        **{f"agg_{s}": agg[s] for s in stages},
        **{f"ins_{s}": curves[s] for s in stages},
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Fig. 2-style aggregate TAD + insulation. "
            "TADs called on --ref-stage (default 8cell); IS profiles for all stages "
            "use those fixed boundaries."
        )
    )
    p.add_argument(
        "--npy-root",
        default="/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/full_triplets/output/config_25k_64",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "aggregate_tad_plots",
            "mouse_embryo_pred_fig2",
        ),
    )
    p.add_argument("--resolution", type=int, default=25_000)
    p.add_argument("--method", default="pred")
    p.add_argument("--stage-prefix", default="mouse_embryo_development")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["early2_cell", "late2_cell", "8cell"],
        help="Stages to plot (ordered)",
    )
    p.add_argument(
        "--ref-stage",
        default="8cell",
        help="Stage used ONLY to call TADs / define boundaries (Fig. 2 ICM→8cell)",
    )
    p.add_argument("--chroms", nargs="+", default=["10", "15", "19"])
    p.add_argument("--bin-start", type=int, default=None)
    p.add_argument("--bin-end", type=int, default=None)
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--ins-window", type=int, default=10)
    p.add_argument(
        "--flank-tad-frac",
        type=float,
        default=0.5,
        help="Flank beyond each TAD edge in units of TAD length (0.5 → 2×TAD window, Fig. 2)",
    )
    p.add_argument("--n-profile", type=int, default=101)
    p.add_argument(
        "--min-prominence",
        type=float,
        default=0.25,
        help="Min prominence of log2-IS dips when calling boundaries",
    )
    p.add_argument("--min-distance-bins", type=int, default=8)
    p.add_argument("--min-tad-bins", type=int, default=4)
    p.add_argument("--max-tad-bins", type=int, default=200)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for chrom in args.chroms:
        run_chrom(
            chrom=chrom,
            stages=args.stages,
            ref_stage=args.ref_stage,
            stage_prefix=args.stage_prefix,
            npy_root=args.npy_root,
            resolution=args.resolution,
            method=args.method,
            grid=args.grid,
            ins_window=args.ins_window,
            flank_tad_frac=args.flank_tad_frac,
            n_profile=args.n_profile,
            out_dir=args.out_dir,
            dpi=args.dpi,
            seed=args.seed,
            bin_start=args.bin_start,
            bin_end=args.bin_end,
            min_prominence=args.min_prominence,
            min_distance_bins=args.min_distance_bins,
            min_tad_bins=args.min_tad_bins,
            max_tad_bins=args.max_tad_bins,
        )


if __name__ == "__main__":
    main()
