import numpy as np
import pandas as pd
import warnings

def vN_entropy(M, SUB_M_SIZE_FIX, CHRSPLIT, PHI_MAX, phi, BIN_TABLE) -> float:
    S = []
    BIN_TABLE_NEW = []
    N = M.shape[0]

    if np.isnan(SUB_M_SIZE_FIX):
        SUB_M_SIZE = np.round(N / CHRSPLIT)
    else:
        SUB_M_SIZE = SUB_M_SIZE_FIX

    SUB_M_SIZE = int(SUB_M_SIZE)
    WN = 1 + np.floor((N - SUB_M_SIZE) / phi)
    while WN > PHI_MAX:
        phi = phi + 1
        WN = 1 + np.floor((N - SUB_M_SIZE) / phi)

    WN = int(WN)

    R1 = np.array(range(0, N, phi))
    R1 = R1[0:WN]
    R2 = R1 + SUB_M_SIZE - 1
    R2 = R2[0:WN]
    R = np.stack((R1, R2), axis=1).astype(int)


    M = M.astype(float)
    M[M == 0] = np.nan
    M = np.log(M)

    for rr in range(0, WN):
        m = M[R[rr, 0] : R[rr, 1] + 1, R[rr, 0] : R[rr, 1] + 1].copy()

        if np.all(np.isnan(m) | (m == 0)):
            ENT = np.nan
        else:
            mask = np.sum((np.isnan(m) | (m == 0)), axis=1) < (SUB_M_SIZE)
            m = m[mask, :]
            m = m[:, mask]

            m[np.isnan(m)] = np.nanmin(m.flatten())
            P = np.corrcoef(m, rowvar=False)
            if not np.all(np.isnan(P)):
                SDs = np.nanstd(m, axis=0, ddof=0) < np.finfo(float).eps
                P[SDs, :] = 0
                P[:, SDs] = 0
                P[np.isnan(P)] = 0  #
                np.fill_diagonal(P, 1.0)
                rho = P / P.shape[0]
                vals = np.linalg.eigvalsh(rho)
                vals = vals[vals > np.finfo(float).eps]
                vals = np.real(vals)
                ENT = -np.sum(vals * np.log(vals))
            else:
                ENT = np.nan

        S.append(ENT)

        BIN_TABLE_NEW.append(
            [
                BIN_TABLE.iloc[R[rr, 0], BIN_TABLE.columns.get_loc("binNr")],
                BIN_TABLE.iloc[R[rr, 1], BIN_TABLE.columns.get_loc("binNr")],
                BIN_TABLE.iloc[R[rr, 0], BIN_TABLE.columns.get_loc("start")],
                BIN_TABLE.iloc[R[rr, 1], BIN_TABLE.columns.get_loc("end")],
            ]
        )

    S = np.array(S, dtype=np.float64)

    BIN_TABLE_NEW = pd.DataFrame(
        BIN_TABLE_NEW, columns=["binNr1", "binNr2", "start", "end"]
    )

    return S, SUB_M_SIZE, WN, phi, BIN_TABLE_NEW



def get_bin_table(M, resolution=10000, chrom='chr1', start_pos=0) -> pd.DataFrame:

    n_bins = M.shape[0]
    has_contact = ~np.all(np.isnan(M), axis=1) & ~np.all(M == 0, axis=1)
    
    BIN_TABLE = pd.DataFrame({
        'BINS_ALL': range(n_bins),
        'chrs': [chrom] * n_bins,
        'START': [start_pos + i * resolution for i in range(n_bins)],
        'END': [start_pos + (i + 1) * resolution for i in range(n_bins)],
        'binNr': range(n_bins),
        'weights': [1.0] * n_bins,
        'CONTACT': [1.0 if c else np.nan for c in has_contact]
    })

    BIN_TABLE = BIN_TABLE.reset_index()
    BIN_TABLE.rename(
        columns={
            "index": "BINS_ALL",
            "chrom": "chrs",
            "start": "START",
            "end": "END",
            "weight": "weights",
        }
    )

    BIN_TABLE["binNr"] = range(0, len(BIN_TABLE))
    INCLUDE = np.where(~np.all(np.isnan(M), axis=1) & ~np.all(M == 0, axis=1))[0]
    BIN_TABLE["CONTACT"] = np.full((BIN_TABLE.shape[0],), np.nan)
    BIN_TABLE.loc[INCLUDE, "CONTACT"] = 1

    return BIN_TABLE

def get_entropy(matrix: np.ndarray) -> float:
    SUB_M_SIZE_FIX = 16
    CHRSPLIT = None
    PHI_MAX = 1e3
    phi = 1
    INCLUDE = set(range(0, matrix.shape[0]))
    BIN_TABLE = get_bin_table(matrix)
    BIN_TABLE = BIN_TABLE.iloc[list(INCLUDE), :].reset_index(drop=True)
    M = matrix[np.meshgrid(sorted(INCLUDE), sorted(INCLUDE), indexing="ij")]
    S, SUB_M_SIZE, WN, phi, BIN_TABLE_NEW = vN_entropy(M, SUB_M_SIZE_FIX, CHRSPLIT, PHI_MAX, phi, BIN_TABLE)
    return S



def get_similarity(mat1: np.ndarray, mat2: np.ndarray) -> float:
    S1 = get_entropy(mat1)
    S2 = get_entropy(mat2)
    non_nan_idx = ~np.isnan(S1) & ~np.isnan(S2)
    Q = np.corrcoef(S1[non_nan_idx], S2[non_nan_idx])[0, 1]
    return Q