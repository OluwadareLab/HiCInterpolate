import os
import re
import shutil
import subprocess

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

_DEFAULT_RSCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flamingo_main_run.R")
_DEFAULT_R = os.path.expanduser("~/.conda/envs/flamingo/bin/Rscript")
_BASIC_MAX_BINS = 200
_ATOM_RE = re.compile(
    r"^ATOM\s+\d+\s+CA\s+\S+\s+(\S+)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


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


def load_coords_pdb(path: str) -> np.ndarray:
    coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            m = _ATOM_RE.match(line)
            if m:
                coords.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
                continue
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


def scc_pdb(xyz_a: np.ndarray, xyz_b: np.ndarray):
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


def scc_pdb_files(pdb_a: str, pdb_b: str):
    return scc_pdb(load_coords_pdb(pdb_a), load_coords_pdb(pdb_b))


def run_flamingo(
    input_file,
    output_dir,
    bin_size,
    chrom,
    start=None,
    end=None,
    domain_res=1_000_000,
    downsampling_rates=0.75,
    lambda_coef=10.0,
    max_dist=0.01,
    n_thread=4,
    max_iter=500,
    alpha=-0.25,
    rscript_bin=None,
    keep_sparse=False,
    force_large=False,
):
    os.makedirs(output_dir, exist_ok=True)
    rscript = rscript_bin or (_DEFAULT_R if os.path.isfile(_DEFAULT_R) else "Rscript")
    if shutil.which(rscript) is None and not os.path.isfile(rscript):
        raise FileNotFoundError(f"Rscript not found: {rscript}")

    chrom_name = chrom if str(chrom).lower().startswith("chr") else f"chr{chrom}"
    mat = load_dense_matrix(input_file)
    if start is not None and end is not None:
        mat = mat[start:end, start:end]
    n = mat.shape[0]
    use_basic = (not force_large) and n <= _BASIC_MAX_BINS
    temps = []

    if use_basic:
        if_path = os.path.join(output_dir, "input_if.txt")
        np.savetxt(if_path, mat, fmt="%.6f")
        temps.append(if_path)
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

    subprocess.run(cmd, check=True)

    coords_tsv = os.path.join(output_dir, "flamingo_coords.tsv")
    if not os.path.isfile(coords_tsv):
        raise RuntimeError(f"Missing FLAMINGO coords: {coords_tsv}")
    xyz = coords_tsv_to_xyz(coords_tsv)
    pdb_path = os.path.join(output_dir, "flamingo_structure.pdb")
    write_pdb(xyz, pdb_path)

    if not keep_sparse:
        for p in temps:
            if os.path.isfile(p):
                os.remove(p)
