#!/usr/bin/env python3
"""3D chromatin structure plots from FLAMINGO PDBs via PyMOL.

Discovers flamingo_structure.pdb under --input_root, groups by
(out_tag, chrom, region), and renders each method as spheres + connecting
trace with rainbow coloring. No alignment or coordinate preprocessing.

Methods: y (GT), pred (Ours), 4dmax, linear, of.
Writes per-structure PNGs (300 dpi) plus a labeled side-by-side panel per group.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

DPI = 300
IMG_IN = 6.67  # inches → ~2000 px @ 300 dpi
IMG_PX = int(round(IMG_IN * DPI))

METHODS = ("y", "pred", "4dmax", "linear", "of")
METHOD_LABELS = {
    "y": "Ground truth",
    "pred": "Ours",
    "4dmax": "4DMax",
    "linear": "Linear",
    "of": "Optical Flow",
}
METHOD_SCC_COLS = {
    "pred": ("ours", "Ours"),
    "4dmax": ("4DMax",),
    "linear": ("Linear",),
    "of": ("Optical Flow",),
}
_KNOWN_SAMPLE_SUBSAMPLE = (
    ("cerebellar_granule_neuron", "control"),
    ("embryo", "development"),
    ("dmso", "control"),
    ("dtag", "v1"),
)

_DIR_RE = re.compile(
    r"^(?P<out_tag>.+)_(?P<chrom>\d+)_(?P<method>y|pred|4dmax|linear|of)$"
)

DEFAULT_INPUT = (
    "/home/hc0783.unt.ad.unt.edu/workspace/hicinterpolate/"
    "datasets/timeseries/full_triplets/output/flamingo_user"
)
DEFAULT_OUTPUT = os.path.join(DEFAULT_INPUT, "structure_plots")
DEFAULT_SCC_CSV = os.path.join(DEFAULT_INPUT, "flamingo_scc_2500_4000.csv")
DEFAULT_PYMOL_LICENSE = (
    "/home/hc0783.unt.ad.unt.edu/pymol-edu-license.lic"
)

SPHERE_SCALE = 1.2
TRACE_RADIUS = 0.8
ZOOM_PAD_FRAC = 0.06  # small framing vs bounding-box diagonal
ZOOM_PAD_MIN = 6.0  # Angstroms; just enough for sphere radii
BG_COLOR = "white"

# (sample, subsample, timestamp, chrom, region) -> {method: score}
SccIndex = Dict[Tuple[str, str, str, str, str], Dict[str, float]]


def parse_out_tag_meta(out_tag: str) -> Tuple[str, str, str]:
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


def region_dir_to_csv(region: str) -> str:
    # region_2500_4000 -> 2500-4000
    parts = region.split("_")
    if len(parts) >= 3 and parts[0] == "region":
        return f"{parts[1]}-{parts[2]}"
    return region.replace("_", "-")


def load_scc_index(path: str) -> SccIndex:
    index: SccIndex = {}
    if not path or not os.path.isfile(path):
        return index
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (
                row["sample"],
                row.get("subsample", ""),
                row.get("timestamp", ""),
                str(row["chromosome"]).lstrip("chr"),
                row.get("region", ""),
            )
            scores: Dict[str, float] = {}
            for method, cols in METHOD_SCC_COLS.items():
                for col in cols:
                    raw = row.get(col, "")
                    if raw is None or raw == "":
                        continue
                    try:
                        scores[method] = float(raw)
                        break
                    except ValueError:
                        continue
            index[key] = scores
    return index


def scc_for_group(
    scc: SccIndex, out_tag: str, chrom: str, region: str
) -> Dict[str, float]:
    sample, subsample, timestamp = parse_out_tag_meta(out_tag)
    key = (sample, subsample, timestamp, str(chrom).lstrip("chr"), region_dir_to_csv(region))
    return scc.get(key, {})


def panel_label(method: str, scores: Dict[str, float]) -> str:
    name = METHOD_LABELS[method]
    if method == "y":
        return name
    score = scores.get(method)
    if score is None:
        return name
    return f"{name} ({score:.3f})"


def discover_groups(root: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """(out_tag, chrom, region) -> {method: pdb_path}."""
    groups: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if not os.path.isdir(root):
        return groups
    for name in sorted(os.listdir(root)):
        m = _DIR_RE.match(name)
        if not m:
            continue
        out_tag = m.group("out_tag")
        chrom = m.group("chrom")
        method = m.group("method")
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        for entry in sorted(os.listdir(run_dir)):
            if not entry.startswith("region_"):
                continue
            pdb = os.path.join(run_dir, entry, "flamingo_structure.pdb")
            if not os.path.isfile(pdb):
                continue
            key = (out_tag, chrom, entry)
            groups.setdefault(key, {})[method] = pdb
    return groups


def find_pymol(explicit: Optional[str] = None) -> str:
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        which = shutil.which(explicit)
        if which:
            return which
        raise FileNotFoundError(f"PyMOL not found: {explicit}")
    for cand in ("pymol", "PyMOL"):
        path = shutil.which(cand)
        if path:
            return path
    for path in (
        "/opt/miniconda3/envs/hicinterpolate/bin/pymol",
        os.path.expanduser("~/miniconda3/envs/hicinterpolate/bin/pymol"),
    ):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "PyMOL not found. Install pymol or pass --pymol /path/to/pymol"
    )


def write_pymol_script(
    pdb_path: str,
    png_path: str,
    script_path: str,
    license_path: Optional[str] = None,
) -> None:
    """Spheres + connecting trace, rainbow, 300 dpi PNG. No alignment."""
    pdb_r = repr(os.path.abspath(pdb_path))
    png_r = repr(os.path.abspath(png_path))
    lic = license_path or DEFAULT_PYMOL_LICENSE
    lic_r = repr(os.path.abspath(lic)) if lic else "None"
    body = f"""from pymol import cmd, licensing
import math
import sys

cmd.reinitialize()

_lic = {lic_r}
if _lic:
    _info = licensing.check_license_file(_lic)
    if not _info.is_valid():
        sys.stderr.write("PyMOL license check failed: " + repr(_info._info) + "\\n")
        sys.exit(1)

cmd.load({pdb_r}, "struct")
cmd.hide("everything", "struct")
cmd.show("spheres", "struct")
cmd.show("sticks", "struct")
cmd.set("sphere_scale", {SPHERE_SCALE})
cmd.set("stick_radius", {TRACE_RADIUS})
cmd.spectrum("count", "rainbow", "struct")
cmd.bg_color("{BG_COLOR}")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("orthoscopic", 1)
cmd.set("ray_shadows", 0)
cmd.orient("struct")
_ext = cmd.get_extent("struct")
_dx = _ext[1][0] - _ext[0][0]
_dy = _ext[1][1] - _ext[0][1]
_dz = _ext[1][2] - _ext[0][2]
_diag = math.sqrt(_dx * _dx + _dy * _dy + _dz * _dz)
_pad = max({ZOOM_PAD_MIN}, {ZOOM_PAD_FRAC} * _diag) + {SPHERE_SCALE} * 3.0
cmd.zoom("struct", buffer=_pad, complete=1)
cmd.clip("atoms", max(5.0, _pad), "struct")
cmd.png({png_r}, width={IMG_PX}, height={IMG_PX}, dpi={DPI}, ray=1)
cmd.quit()
"""
    with open(script_path, "w") as fh:
        fh.write(body)


def ensure_pymol_license(pymol: str, license_path: str) -> bool:
    """Install license into ~/.pymol so cold starts find it (env var alone is ignored)."""
    if not license_path or not os.path.isfile(license_path):
        return False
    user_lic = os.path.expanduser("~/.pymol/license.lic")
    if os.path.isfile(user_lic) and os.path.getsize(user_lic) > 0:
        # still verify it works
        pass
    script = None
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_lic.py", delete=False, prefix="pymol_lic_"
    ) as fh:
        script = fh.name
        fh.write(
            "from pymol import licensing, cmd\n"
            f"info = licensing.install_license_file({os.path.abspath(license_path)!r})\n"
            "print('LICENSE', info._info)\n"
            "raise SystemExit(0 if info.is_valid() else 1)\n"
        )
    try:
        env = _pymol_env(pymol, license_path)
        proc = subprocess.run(
            [pymol, "-cq", script], capture_output=True, text=True, env=env
        )
        ok = proc.returncode == 0 and os.path.isfile(user_lic)
        if not ok:
            sys.stderr.write((proc.stdout or "") + (proc.stderr or "") + "\n")
        return ok
    finally:
        try:
            os.remove(script)
        except OSError:
            pass


def _pymol_env(pymol: str, license_path: Optional[str] = None) -> dict:
    """Ensure pymol's lib dir is on LD_LIBRARY_PATH (conda installs)."""
    env = os.environ.copy()
    root = os.path.dirname(os.path.dirname(os.path.abspath(pymol)))
    lib = os.path.join(root, "lib")
    if os.path.isdir(lib):
        env["LD_LIBRARY_PATH"] = lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    lic = license_path or DEFAULT_PYMOL_LICENSE
    if lic and os.path.isfile(lic):
        # used by some PyMOL/rigimol paths; primary activation is check_license_file
        env["PYMOL_LICENSE_FILE"] = os.path.abspath(lic)
    return env


def render_pymol(
    pymol: str,
    script_path: str,
    png_path: str,
    license_path: Optional[str] = None,
) -> None:
    if os.path.isfile(png_path):
        os.remove(png_path)
    env = _pymol_env(pymol, license_path)
    cmds = [
        [pymol, "-cq", script_path],
        [pymol, "-c", script_path],
    ]
    xvfb = shutil.which("xvfb-run")
    if xvfb:
        cmds = [[xvfb, "-a"] + c for c in cmds] + cmds

    last = None
    logs: List[str] = []
    for cmd in cmds:
        last = subprocess.run(cmd, capture_output=True, text=True, env=env)
        blob = ((last.stdout or "") + (last.stderr or "")).strip()
        if blob:
            logs.append(blob)
        if _png_ok(png_path):
            return
    detail = "\n".join(logs[-2:]) if logs else ""
    if detail:
        sys.stderr.write(detail + "\n")
    hint = ""
    if "libCatch2" in detail or "undefined symbol" in detail:
        hint = (
            "\nHint: PyMOL ABI mismatch — in the pymol conda env run:\n"
            "  conda install -c conda-forge 'catch2=3.13.0'"
        )
    raise RuntimeError(
        f"PyMOL failed to write {png_path} "
        f"(exit {getattr(last, 'returncode', '?')}){hint}"
    )


def _png_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def crop_whitespace(png_path: str, pad_frac: float = 0.03, white: int = 250) -> None:
    """Crop near-white margins; keep a small pad around content."""
    from PIL import Image
    import numpy as np

    im = Image.open(png_path).convert("RGBA")
    arr = np.asarray(im)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    content = (rgb.min(axis=2) < white) & (alpha > 0)
    if not content.any():
        return
    ys, xs = np.where(content)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = content.shape
    pad = max(2, int(round(pad_frac * max(y1 - y0 + 1, x1 - x0 + 1))))
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h - 1, y1 + pad)
    x1 = min(w - 1, x1 + pad)
    cropped = im.crop((x0, y0, x1 + 1, y1 + 1))
    # flatten onto white (drop alpha fringe)
    out = Image.new("RGB", cropped.size, (255, 255, 255))
    out.paste(cropped, mask=cropped.split()[-1])
    out.save(png_path, dpi=(DPI, DPI))


def _fit_square(img, size: int = 800, bg=(255, 255, 255)):
    """Fit RGB/RGBA image into a fixed square canvas, centered, aspect preserved."""
    from PIL import Image
    import numpy as np

    if hasattr(img, "convert"):
        src = img.convert("RGBA")
    else:
        arr = np.asarray(img)
        if arr.dtype.kind == "f":
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            src = Image.fromarray(arr).convert("RGBA")
        elif arr.shape[2] == 4:
            src = Image.fromarray(arr.astype(np.uint8), "RGBA")
        else:
            src = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")

    w, h = src.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    resized = src.resize((nw, nh), resample)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2), mask=resized.split()[-1])
    return canvas


def compose_panel(
    pngs: Dict[str, str],
    out_path: str,
    scores: Optional[Dict[str, float]] = None,
) -> None:
    """Equal square cells; titles (method + SCC) aligned across columns."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    methods = [m for m in METHODS if m in pngs and _png_ok(pngs[m])]
    if not methods:
        return
    scores = scores or {}

    cell = 900
    gap = 24
    title_h = 70
    pad_top = 8
    pad_side = 8
    n = len(methods)
    width = pad_side * 2 + n * cell + (n - 1) * gap
    height = pad_top + title_h + cell + pad_side
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56
        )
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 56)
        except OSError:
            font = ImageFont.load_default()

    for i, method in enumerate(methods):
        x0 = pad_side + i * (cell + gap)
        tile = _fit_square(Image.open(pngs[method]), size=cell)
        panel.paste(tile, (x0, pad_top + title_h))
        label = panel_label(method, scores)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (cell - tw) // 2
        ty = pad_top + (title_h - th) // 2
        draw.text((tx, ty), label, fill=(30, 30, 30), font=font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    panel.save(out_path, dpi=(DPI, DPI))


def process_group(
    key: Tuple[str, str, str],
    pdb_map: Dict[str, str],
    output_dir: str,
    pymol: str,
    skip_panel: bool,
    want_methods: Optional[set] = None,
    scc: Optional[SccIndex] = None,
    license_path: Optional[str] = None,
) -> List[str]:
    out_tag, chrom, region = key
    written: List[str] = []
    render_set = want_methods or set(METHODS)
    scores = scc_for_group(scc or {}, out_tag, chrom, region)

    slug = f"{out_tag}_chr{chrom}_{region}"
    group_dir = os.path.join(output_dir, slug)
    os.makedirs(group_dir, exist_ok=True)

    pngs: Dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="flamingo_pymol_") as tmp:
        for method in METHODS:
            if method not in pdb_map or method not in render_set:
                continue
            pdb_path = pdb_map[method]
            png_path = os.path.join(group_dir, f"{method}.png")
            script = os.path.join(tmp, f"{method}.pml.py")
            try:
                write_pymol_script(
                    pdb_path, png_path, script, license_path=license_path
                )
                render_pymol(pymol, script, png_path, license_path=license_path)
            except Exception as ex:
                print(f"FAIL render {slug} {method}: {ex}", flush=True)
                continue
            if not _png_ok(png_path):
                print(f"FAIL missing png {png_path}", flush=True)
                continue
            try:
                crop_whitespace(png_path)
            except Exception as ex:
                print(f"WARN crop {slug} {method}: {ex}", flush=True)
            pngs[method] = png_path
            written.append(png_path)
            print(f"Wrote {png_path}", flush=True)

    if not skip_panel and pngs:
        panel = os.path.join(group_dir, "panel.png")
        compose_panel(pngs, panel, scores)
        if _png_ok(panel):
            written.append(panel)
            print(f"Wrote {panel}", flush=True)
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", default=DEFAULT_INPUT)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    p.add_argument("--scc_csv", default=DEFAULT_SCC_CSV, help="SCC scores CSV")
    p.add_argument("--pymol", default=None, help="Path to pymol binary")
    p.add_argument(
        "--pymol_license",
        default=DEFAULT_PYMOL_LICENSE,
        help="Path to PyMOL license.lic",
    )
    p.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated methods to render",
    )
    p.add_argument(
        "--chrom",
        default=None,
        help="Optional chromosome filter, e.g. 10 or 10,15",
    )
    p.add_argument("--limit", type=int, default=None, help="Render at most N groups")
    p.add_argument("--skip_panel", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        pymol = find_pymol(args.pymol)
    except FileNotFoundError as ex:
        print(f"ERROR: {ex}", flush=True)
        return 1
    print(f"PyMOL: {pymol}", flush=True)
    if args.pymol_license and os.path.isfile(args.pymol_license):
        print(f"License: {args.pymol_license}", flush=True)
        if ensure_pymol_license(pymol, args.pymol_license):
            print("License installed → ~/.pymol/license.lic", flush=True)
        else:
            print("WARN: failed to install PyMOL license to ~/.pymol", flush=True)
    else:
        print(f"WARN: PyMOL license not found: {args.pymol_license}", flush=True)

    scc = load_scc_index(args.scc_csv)
    if scc:
        print(f"SCC: {args.scc_csv} ({len(scc)} rows)", flush=True)
    else:
        print(f"WARN: no SCC scores from {args.scc_csv}", flush=True)

    want_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    chrom_filter = None
    if args.chrom:
        chrom_filter = {
            c.strip().lstrip("chr") for c in args.chrom.split(",") if c.strip()
        }

    groups = discover_groups(args.input_root)
    if not groups:
        print(f"No PDB groups under {args.input_root}", flush=True)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    keys = sorted(groups.keys())
    if chrom_filter:
        keys = [k for k in keys if k[1] in chrom_filter]
    if args.limit is not None:
        keys = keys[: max(0, args.limit)]

    n_ok = 0
    for key in keys:
        written = process_group(
            key,
            dict(groups[key]),
            args.output_dir,
            pymol=pymol,
            skip_panel=args.skip_panel,
            want_methods=want_methods,
            scc=scc,
            license_path=args.pymol_license,
        )
        if written:
            n_ok += 1

    print(f"Done. Rendered {n_ok}/{len(keys)} groups → {args.output_dir}", flush=True)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
