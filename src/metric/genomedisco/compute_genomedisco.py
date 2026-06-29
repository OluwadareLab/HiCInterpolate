import copy
import numpy as np
import torch
from src.metric.genomedisco import data_operations
from src.metric.genomedisco.comparison_types.disco_random_walks import DiscoRandomWalks
from src.metric.genomedisco import processing


def compute_reproducibility_from_tensor(pred: torch.Tensor, target: torch.Tensor):
    c = 0
    for b in range(pred.shape[0]):
        pred_mat = pred[b, c, :, :].detach().cpu().numpy()
        target_mat = target[b, c, :, :].detach().cpu().numpy()
        reproducibility = compute_reproducibility(pred_mat, target_mat)
        if b == 0:
            reproducibility_list = [reproducibility]
        else:
            reproducibility_list.append(reproducibility)
    return np.array(reproducibility_list).mean()


def compute_reproducibility(pred: np.ndarray, target: np.ndarray):
    # try:
    #     assert pred.shape == target.shape
    # except Exception as e:
    #     print(e)
    #     return np.nan

    pred_csr = processing.construct_csr_matrix_from_data_and_nodes(pred)
    target_csr = processing.construct_csr_matrix_from_data_and_nodes(target)

    stats = {}
    stats['mat1'] = {}
    stats['mat2'] = {}
    stats['mat1']['depth'] = pred_csr.sum()
    stats['mat2']['depth'] = target_csr.sum()

    m1_subsample = copy.deepcopy(pred_csr)
    m2_subsample = copy.deepcopy(target_csr)

    if stats['mat1']['depth'] >= stats['mat2']['depth']:
        m_subsample = copy.deepcopy(target_csr)
    if stats['mat1']['depth'] < stats['mat2']['depth']:
        m_subsample = copy.deepcopy(pred_csr)
    desired_depth = m_subsample.sum()

    if pred_csr.sum() > desired_depth:
        m1_subsample = data_operations.subsample_to_depth(
            pred_csr, desired_depth)
    if target_csr.sum() > desired_depth:
        m2_subsample = data_operations.subsample_to_depth(
            target_csr, desired_depth)

    stats['mat1']['subsampled_depth'] = m1_subsample.sum()
    stats['mat2']['subsampled_depth'] = m2_subsample.sum()

    m1_norm = data_operations.process_matrix(m1_subsample, 'sqrtvc')
    m2_norm = data_operations.process_matrix(m2_subsample, 'sqrtvc')

    comparer = DiscoRandomWalks()
    reproducibility = comparer.compute_reproducibility(m1_norm, m2_norm)

    return reproducibility


def get_dd_diff(m1dd, m2dd):
    d = 0.0
    k = set(m1dd.keys()).union(set(m2dd.keys()))
    for key in k:
        if key in m1dd:
            m1val = m1dd[key]
        else:
            m1val = 0.0
        if key in m2dd:
            m2val = m2dd[key]
        else:
            m2val = 0.0
        d += abs(m1val-m2val)
    return d
