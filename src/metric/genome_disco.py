import numpy as np
import copy
from sklearn import metrics
import scipy.sparse as sps

# Code taken from https://github.com/kundajelab/genomedisco


def to_transition(mtogether):
    sums = mtogether.sum(axis=1)
    sums[sums == 0.0] = 1.0
    D = sps.spdiags(1.0 / sums.flatten(),
                    [0], mtogether.shape[0], mtogether.shape[1], format='csr')
    return D.dot(mtogether)


def compute_reproducibility(m1_csr, m2_csr, transition, tmax=3, tmin=3):
    m1up = m1_csr
    m1down = m1up.transpose()
    m1 = m1up + m1down

    m2up = m2_csr
    m2down = m2up.transpose()
    m2 = m2up + m2down

    if transition:
        m1 = to_transition(m1)
        m2 = to_transition(m2)

    rowsums_1 = m1.sum(axis=1)
    nonzero_1 = [i for i in range(rowsums_1.shape[0]) if rowsums_1[i] > 0.0]
    rowsums_2 = m2.sum(axis=1)
    nonzero_2 = [i for i in range(rowsums_2.shape[0]) if rowsums_2[i] > 0.0]
    nonzero_total = len(list(set(nonzero_1).union(set(nonzero_2))))
    nonzero_total = 0.5 * (1.0 * len(list(set(nonzero_1))) +
                           1.0 * len(list(set(nonzero_2))))

    scores = []
    if True:
        diff_vector = np.zeros((m1.shape[0], 1))
        for t in range(1, tmax + 1): 
            extra_text = ' (not included in score calculation)'
            if t == 1:
                rw1 = copy.deepcopy(m1)
                rw2 = copy.deepcopy(m2)

            else:
                rw1 = rw1.dot(m1)
                rw2 = rw2.dot(m2)

            if t >= tmin:
                diff = abs(rw1 - rw2).sum()
                scores.append(1.0 * float(diff) / float(nonzero_total))
                extra_text = ' | score=' + \
                    str('{:.3f}'.format(1.0 - float(diff) / float(nonzero_total)))

    ts = range(tmin, tmax + 1)
    denom = len(ts) - 1
    if tmin == tmax:
        auc = scores[0]

        if 2 < auc:
            auc = 2

        elif 0 <= auc <= 2:
            auc = auc

    else:
        auc = metrics.auc(range(len(ts)), scores) / denom

    reproducibility = 1 - auc
    return reproducibility
