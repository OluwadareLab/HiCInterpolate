from typing import List
import cupy as cp
import torch
import numpy as np
import _4DMax.Utils.util as ut
import _4DMax.Utils.movement as mv
import _4DMax.Utils.likelihood as li


def get_sparse_matrix(matrix):
    rows, cols = np.where(matrix > 0)
    values = matrix[rows, cols]

    sparse_matrix = np.column_stack((rows, cols, values))
    return sparse_matrix


def normalize(mat, is_log=False, is_min_max=False):
    if is_log:
        mat = np.log1p(mat)

    if is_min_max:
        min_val = np.min(mat)
        max_val = np.max(mat)
        denom = max_val - min_val
        if denom == 0:
            fill_value = 1.0 if min_val > 0 else 0.0
            mat = np.full_like(mat, fill_value, dtype=np.float32)
        else:
            mat = (mat - min_val) / denom

    return mat


def run_4dmax(timeframe: List[torch.Tensor], patch_size=64):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    np.random.seed(42)

    eta = 100
    alpha = 0.6
    lr = 0.0001
    epochs = 300

    start_t = 0
    end_t = 2
    step = 1
    taos = np.array([0, 1])
    ts = np.linspace(start_t, end_t, step)

    map_tao = {}
    row_tao = {}
    col_tao = {}
    hic_dist_tao = {}
    ifs_tao = {}
    n_tao = {}
    n_min_tao = {}

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    sparse_matrix_size = patch_size * patch_size
    for key, val in enumerate(taos):
        DEVICE = timeframe[val].device
        dense_matrix = timeframe[val].squeeze().cpu().numpy()
        sparse_matrix = get_sparse_matrix(dense_matrix)

        map_tao[val] = sparse_matrix
        row_tao[val] = (map_tao[val][:, 0].astype(int)).astype(int)
        col_tao[val] = (map_tao[val][:, 1].astype(int)).astype(int)
        ifs_tao[val] = map_tao[val][:, 2]
        hic_dist_tao[val] = ut.if2dist(ifs_tao[val], alpha)
        n_tao[val] = np.max((row_tao[val], col_tao[val]))
        n_min_tao[val] = np.min((row_tao[val], col_tao[val]))

    n_max = n_tao[list(n_tao.keys())[0]]
    n_max = np.max(list(n_tao.values()))
    n_min = np.min(list(n_min_tao.values()))

    for key, val in enumerate(taos):
        row_tao[val] = row_tao[val] - n_min_tao[val]
        col_tao[val] = col_tao[val] - n_min_tao[val]
    struc_t = np.random.rand(ts.shape[0], n_max+1-n_min, 3)

    GPU = True
    if GPU:
        for i in hic_dist_tao.keys():
            hic_dist_tao[i] = cp.array(hic_dist_tao[i])
        struc_t = cp.array(struc_t)
        ts = cp.array(ts)
        taos = cp.array(taos)

        for e in range(0, epochs):
            likelihood_loss = li.likelihoodlossGPU(
                hic_dist_tao, row_tao, col_tao, struc_t, ts, taos, n_max, n_min)
            movement_loss = mv.movementLossGPU(struc_t)
            struc_t -= lr*(likelihood_loss+(eta*movement_loss))

    else:
        for e in range(0, epochs):
            likelihood_loss = li.likelihoodloss(
                hic_dist_tao, row_tao, col_tao, struc_t, ts, taos, n_max, n_min)
            movement_loss = mv.movementLoss(struc_t)
            struc_t -= lr*(likelihood_loss+(eta*movement_loss))

    pred = ut.loadStrucAtTimeAsMat(struc_t, 0)
    pr, pc = pred.shape
    if pr != pc:
        print(
            f"[Warning] Skipping due to non-square predicted matrix")
        count += 1
        return np.nan

    pad_amount = patch_size - pr
    if pad_amount > 0:
        print(
            f"[Warning] Skipping due to shape mismatch between predicted {pred.shape} and ground truth {timeframe[0].shape}")
        # pred = np.pad(pred, ((0, pad_amount), (0, pad_amount)),
        #               mode='constant', constant_values=0)
        return np.nan

    pred_tensor = torch.tensor(pred).unsqueeze(
        0).unsqueeze(0).float().to(DEVICE)

    return pred_tensor
