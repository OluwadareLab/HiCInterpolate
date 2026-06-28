import copy
from src.metric.genomedisco import data_operations
from src.metric.genomedisco.comparison_types.disco_random_walks import DiscoRandomWalks
from src.metric.genomedisco import processing


def compute_reproducibility(matrix1, matrix2):

    m1 = processing.construct_csr_matrix_from_data_and_nodes(matrix1)
    m2 = processing.construct_csr_matrix_from_data_and_nodes(matrix2)

    stats = {}
    stats['mat1'] = {}
    stats['mat2'] = {}
    stats['mat1']['depth'] = m1.sum()
    stats['mat2']['depth'] = m2.sum()

    m1_subsample = copy.deepcopy(m1)
    m2_subsample = copy.deepcopy(m2)

    if stats['mat1']['depth'] >= stats['mat2']['depth']:
        m_subsample = copy.deepcopy(m2)
    if stats['mat1']['depth'] < stats['mat2']['depth']:
        m_subsample = copy.deepcopy(m1)
    desired_depth = m_subsample.sum()

    if m1.sum() > desired_depth:
        m1_subsample = data_operations.subsample_to_depth(
            m1, desired_depth)
    if m2.sum() > desired_depth:
        m2_subsample = data_operations.subsample_to_depth(
            m2, desired_depth)

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
