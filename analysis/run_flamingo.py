#!/usr/bin/env python3
"""Run FLAMINGOr (wangjr03/FLAMINGO) on .hic / .mcool / dense nxn Hi-C."""

import argparse
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RSCRIPT = os.path.join(_SCRIPT_DIR, "flamingo_main_run.R")
_DEFAULT_R = os.path.expanduser("~/.conda/envs/flamingo/bin/Rscript")
_BASIC_MAX_BINS = 200


def load_dense_matrix(matrix_file: str) -> np.ndarray:
    _, ext = os.path.splitext(matrix_file)
    ext = ext.lower()
    if ext == ".npy":
        mat = np.load(matrix_file)
    elif ext in {".txt", ".tsv", ".csv"}:
        delim = "," if ext == ".csv" else None
        mat = np.loadtxt(matrix_file, delimiter=delim)
    else:
        raise ValueError(f"Unsupported dense matrix format '{ext}'")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {mat.shape}")
    return mat.astype(np.float64, copy=False)


def normalize_by_ref(mat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale mat to [0, 1] using ref min/max (same scale for all methods)."""
    ref = np.asarray(ref, dtype=np.float64)
    finite = ref[np.isfinite(ref)]
    if finite.size == 0:
        raise ValueError("norm_ref has no finite values")
    ymin = float(finite.min())
    ymax = float(finite.max())
    scale = ymax - ymin
    if scale <= 0:
        scale = 1.0
    out = (mat.astype(np.float64, copy=False) - ymin) / scale
    np.clip(out, 0.0, 1.0, out=out)
    print(f"Normalized to [0,1] with y-scale ymin={ymin:.6g} ymax={ymax:.6g}", flush=True)
    return out


def write_pdb(positions: np.ndarray, pdb_file: str, scale: float = 100.0) -> None:
    pos = np.asarray(positions, dtype=np.float64) * scale
    with open(pdb_file, "w") as o_file:
        o_file.write("\n")
        n = len(pos)
        for i in range(1, n + 1):
            col2 = f"{i:5d}"
            col4 = f"B{i}"
            col4 = col4 + " " * (6 - len(col4))
            col5 = f"{pos[i - 1][0]:8.3f}"
            col6 = f"{pos[i - 1][1]:8.3f}"
            col7 = f"{pos[i - 1][2]:8.3f}"
            o_file.write(
                f"ATOM  {col2}   CA MET {col4}   {col5}{col6}{col7}  0.20 10.00\n"
            )
        for i in range(1, n + 1):
            j = i + 1
            if j > n:
                break
            o_file.write(f"CONECT{i:5d}{j:5d}\n")
        o_file.write("END")


def dense_to_rao_sparse(mat: np.ndarray, bin_size: int, sparse_path: str, norm_path: str) -> int:
    n = mat.shape[0]
    rows, cols = np.triu_indices(n)
    vals = mat[rows, cols]
    mask = np.isfinite(vals) & (vals > 0)
    rows, cols, vals = rows[mask], cols[mask], vals[mask]
    bp1 = rows.astype(np.int64) * bin_size
    bp2 = cols.astype(np.int64) * bin_size
    np.savetxt(sparse_path, np.column_stack([bp1, bp2, vals]), fmt=["%d", "%d", "%.6f"])
    np.savetxt(norm_path, np.ones(n, dtype=np.float64), fmt="%.6f")
    return n


def aggregate_matrix(mat: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mat
    n = (mat.shape[0] // factor) * factor
    if n == 0:
        raise ValueError(f"Matrix too small ({mat.shape[0]}) for domain factor {factor}")
    cropped = mat[:n, :n]
    m = n // factor
    return cropped.reshape(m, factor, m, factor).sum(axis=(1, 3))


def detect_input_kind(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".hic":
        return "hic"
    if ext in {".mcool", ".cool"}:
        return "mcool"
    if ext == ".npy":
        return "dense"
    if ext in {".txt", ".tsv", ".csv"}:
        sample = np.loadtxt(path, delimiter="," if ext == ".csv" else None, max_rows=8)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        if sample.shape[1] == 3:
            return "sparse"
        return "dense"
    raise ValueError(f"Unrecognized input: {path}")


def coords_tsv_to_xyz(tsv_path: str) -> np.ndarray:
    data = np.genfromtxt(tsv_path, names=True, dtype=None, encoding=None)
    names = [n.lower() for n in data.dtype.names]
    if all(k in names for k in ("x", "y", "z")):
        xyz = np.column_stack(
            [data[data.dtype.names[names.index(k)]] for k in ("x", "y", "z")]
        )
    else:
        arr = np.genfromtxt(tsv_path, skip_header=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xyz = arr[:, -3:]
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def run_flamingo(
    input_file,
    output_dir,
    bin_size,
    chrom,
    domain_res=1_000_000,
    chr_size=None,
    start=None,
    end=None,
    normalization="NONE",
    downsampling_rates=0.75,
    lambda_coef=10.0,
    max_dist=0.01,
    n_thread=4,
    max_iter=500,
    alpha=-0.25,
    rscript_bin=None,
    keep_sparse=False,
    force_large=False,
    norm_ref=None,
):
    os.makedirs(output_dir, exist_ok=True)
    rscript = rscript_bin or (_DEFAULT_R if os.path.isfile(_DEFAULT_R) else "Rscript")
    if shutil.which(rscript) is None and not os.path.isfile(rscript):
        raise FileNotFoundError(f"Rscript not found: {rscript}")

    chrom_name = chrom if str(chrom).lower().startswith("chr") else f"chr{chrom}"
    kind = detect_input_kind(input_file)
    temps = []

    if kind in {"hic", "mcool"}:
        if chr_size is None:
            raise ValueError("--chr_size is required for .hic / .mcool inputs")
        if start is not None or end is not None:
            raise ValueError("--start/--end only supported for dense matrix inputs")
        cmd = [
            rscript, _DEFAULT_RSCRIPT, "large",
            "hic" if kind == "hic" else "mcool",
            os.path.abspath(input_file),
            os.path.abspath(output_dir),
            str(int(domain_res)),
            str(int(bin_size)),
            str(int(chr_size)),
            chrom_name,
            normalization,
            str(downsampling_rates),
            str(lambda_coef),
            str(max_dist),
            str(int(n_thread)),
            "NA", "NA", "NA",
            str(int(max_iter)),
            str(alpha),
        ]
    elif kind == "sparse":
        raise ValueError(
            "Standalone sparse needs low+high+norms. "
            "Pass dense .npy/.txt (auto-converted) or .hic/.mcool."
        )
    else:
        mat = load_dense_matrix(input_file)
        if start is not None and end is not None:
            mat = mat[start:end, start:end]
        if norm_ref:
            ref = load_dense_matrix(norm_ref)
            if start is not None and end is not None:
                ref = ref[start:end, start:end]
            if ref.shape != mat.shape:
                raise ValueError(
                    f"norm_ref region shape {ref.shape} != input shape {mat.shape}"
                )
            mat = normalize_by_ref(mat, ref)
        n = mat.shape[0]
        use_basic = (not force_large) and n <= _BASIC_MAX_BINS

        if use_basic:
            if_path = os.path.join(output_dir, "input_if.txt")
            np.savetxt(if_path, mat, fmt="%.6f")
            temps.append(if_path)
            print(f"Dense {n}x{n} -> FLAMINGOr basic (reconstruct_structure_worker)", flush=True)
            cmd = [
                rscript, _DEFAULT_RSCRIPT, "basic",
                os.path.abspath(if_path),
                os.path.abspath(output_dir),
                str(downsampling_rates),
                str(lambda_coef),
                str(max_dist),
                str(int(n_thread)),
                str(alpha),
            ]
        else:
            if domain_res % bin_size != 0:
                raise ValueError(
                    f"domain_res ({domain_res}) must be divisible by bin_size ({bin_size})"
                )
            if_path = os.path.join(output_dir, "input_if.bin")
            with open(if_path, "wb") as fh:
                np.asarray([n, n], dtype=np.int32).tofile(fh)
                np.ascontiguousarray(mat, dtype=np.float64).ravel(order="C").tofile(fh)
            temps.append(if_path)
            print(
                f"Dense {n}x{n} full chromosome -> FLAMINGOr dense_large "
                f"(domain_res={domain_res}, frag_res={bin_size})",
                flush=True,
            )
            cmd = [
                rscript, _DEFAULT_RSCRIPT, "dense_large",
                os.path.abspath(if_path),
                os.path.abspath(output_dir),
                str(int(domain_res)),
                str(int(bin_size)),
                chrom_name,
                str(downsampling_rates),
                str(lambda_coef),
                str(max_dist),
                str(int(n_thread)),
                str(int(max_iter)),
                str(alpha),
            ]

    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    coords_tsv = os.path.join(output_dir, "flamingo_coords.tsv")
    if not os.path.isfile(coords_tsv):
        raise RuntimeError(f"Missing FLAMINGO coords: {coords_tsv}")
    xyz = coords_tsv_to_xyz(coords_tsv)
    pdb_path = os.path.join(output_dir, "flamingo_structure.pdb")
    write_pdb(xyz, pdb_path)
    print(f"Wrote {pdb_path} ({len(xyz)} atoms)", flush=True)

    if not keep_sparse:
        for p in temps:
            if os.path.isfile(p):
                os.remove(p)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "--matrix_file", dest="input_file", required=True,
                   help="Input .hic / .mcool / dense .npy|.txt (nxn)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bin_size", type=int, required=True, help="Fragment resolution bp")
    p.add_argument("--chrom", required=True)
    p.add_argument("--domain_res", type=int, default=1_000_000)
    p.add_argument("--chr_size", type=int, default=None, help="Required for .hic/.mcool")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--normalization", default="NONE")
    p.add_argument("--downsampling_rates", type=float, default=0.75)
    p.add_argument("--lambda_coef", type=float, default=10.0)
    p.add_argument("--max_dist", type=float, default=0.01)
    p.add_argument("--n_thread", type=int, default=4)
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--alpha", type=float, default=-0.25)
    p.add_argument("--rscript", dest="rscript_bin", default=None)
    p.add_argument("--keep_sparse", action="store_true")
    p.add_argument("--force_large", action="store_true",
                   help="Force hierarchical flamingo.main_func_large even if n<=200")
    p.add_argument(
        "--norm_ref",
        default=None,
        help="Dense y matrix used to min-max scale input to [0,1] (same scale for all methods)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        run_flamingo(
            input_file=args.input_file,
            output_dir=args.output_dir,
            bin_size=args.bin_size,
            chrom=args.chrom,
            domain_res=args.domain_res,
            chr_size=args.chr_size,
            start=args.start,
            end=args.end,
            normalization=args.normalization,
            downsampling_rates=args.downsampling_rates,
            lambda_coef=args.lambda_coef,
            max_dist=args.max_dist,
            n_thread=args.n_thread,
            max_iter=args.max_iter,
            alpha=args.alpha,
            rscript_bin=args.rscript_bin,
            keep_sparse=args.keep_sparse,
            force_large=args.force_large,
            norm_ref=args.norm_ref,
        )
    except Exception as ex:
        print(f"Exception: {ex}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
