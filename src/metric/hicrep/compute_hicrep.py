from scipy.stats import rankdata
import numpy as np
import scipy.ndimage as ndimage


def fast_mean_filter(mat, h):
    if h == 0:
        return np.asmatrix(mat)
    size = 2 * h + 1

    filtered = ndimage.uniform_filter(
        np.asmatrix(mat), size=size, mode='nearest')
    return filtered


def vstran(d):
    n_rows = d.shape[0]

    perm1 = np.random.permutation(n_rows)
    ranks1 = rankdata(d[perm1, 0], method='ordinal')
    x1r = np.empty_shape = np.zeros(n_rows)
    x1r[perm1] = ranks1

    perm2 = np.random.permutation(n_rows)
    ranks2 = rankdata(d[perm2, 1], method='ordinal')
    x2r = np.zeros(n_rows)
    x2r[perm2] = ranks2

    x1_cdf = x1r / n_rows
    x2_cdf = x2r / n_rows

    new_d = np.column_stack((x1_cdf, x2_cdf))

    return new_d


def compute_hicrep_scc(mat1, mat2, resol=25000, h=1, lbr=0, ubr=25000 * 253):
    if h == 0:
        smt_R1 = np.asarray(mat1)
        smt_R2 = np.asarray(mat2)
    else:
        smt_R1 = np.asarray(fast_mean_filter(mat1, h))
        smt_R2 = np.asarray(fast_mean_filter(mat2, h))

    lb = int(np.floor(lbr / resol))
    ub = int(np.floor(ubr / resol))

    corr_list = []
    wei_list = []

    for dist in range(lb, ub + 1):
        ffd1 = []
        ffd2 = []

        n_cols = smt_R1.shape[1]
        for i in range(n_cols - dist):
            ffd1.append(smt_R1[i + dist, i])
            ffd2.append(smt_R2[i + dist, i])

        ffd1 = np.array(ffd1)
        ffd2 = np.array(ffd2)

        mask = ~((ffd1 == 0) & (ffd2 == 0))
        ffd1_filt = ffd1[mask]
        ffd2_filt = ffd2[mask]

        if len(ffd1_filt) > 0:
            n = len(ffd1_filt)
            ffd = np.column_stack((ffd1_filt, ffd2_filt))
            nd = vstran(ffd)

            if len(np.unique(ffd[:, 0])) != 1 and len(np.unique(ffd[:, 1])) != 1:
                corr_val = np.corrcoef(ffd[:, 0], ffd[:, 1])[0, 1]

                var_nd0 = np.var(nd[:, 0], ddof=1)
                var_nd1 = np.var(nd[:, 1], ddof=1)
                wei_val = np.sqrt(var_nd0 * var_nd1) * n
            else:
                corr_val = np.nan
                wei_val = np.nan
        else:
            corr_val = np.nan
            wei_val = np.nan

        corr_list.append(corr_val)
        wei_list.append(wei_val)

    corr_arr = np.array(corr_list)
    wei_arr = np.array(wei_list)

    valid_mask = ~np.isnan(corr_arr) & ~np.isnan(wei_arr)
    corr = corr_arr[valid_mask]
    wei = wei_arr[valid_mask] + 1e-10

    if len(corr) == 0:
        return np.nan

    scc = np.dot(corr, wei) / np.sum(wei)

    return scc


def get_hicrep_scc(mat1, mat2, resol=25000, patch_size=64, h=1):
    try:
        scc = compute_hicrep_scc(mat1, mat2, resol, h, 0, resol*(patch_size-1))
        return scc
    except Exception as e:
        print(f"Error computing HiCRep SCC: {e}")
        return np.nan


def get_hicrep_scc_from_tensor(mat1, mat2, resol=25000, patch_size=64, h=1):
    c = 0
    hicrep_scc_list = []
    for b in range(mat1.shape[0]):
        m1 = mat1[b, c, :, :].detach().cpu().numpy()
        m2 = mat2[b, c, :, :].detach().cpu().numpy()
        try:
            hicrep_scc = compute_hicrep_scc(m1, m2, resol, h, 0, resol*(patch_size-1))
            hicrep_scc_list.append(hicrep_scc)
        except Exception as e:
            print(f"Error computing HiCRep SCC: {e}")
            hicrep_scc_list.append(np.nan)

    avg_hicrep_scc = np.nanmean(hicrep_scc_list)
    return avg_hicrep_scc
