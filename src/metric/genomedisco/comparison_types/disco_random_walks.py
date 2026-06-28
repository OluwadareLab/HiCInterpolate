import scipy.sparse as sps
from sklearn import metrics
import copy
import numpy as np
import gzip


def to_transition(mtogether):
    sums = mtogether.sum(axis=1)
    sums[sums == 0.0] = 1.0
    D = sps.spdiags(1.0/sums.flatten(),
                    [0], mtogether.get_shape()[0], mtogether.get_shape()[1], format='csr')
    return D.dot(mtogether)


def random_walk(m_input, t):
    return m_input.__pow__(t)


def write_diff_vector_bedfile(diff_vector, nodes, nodes_idx, out_filename):
    out = gzip.open(out_filename, 'w')
    for i in range(diff_vector.shape[0]):
        node_name = nodes_idx[i]
        node_dict = nodes[node_name]
        out.write(str(node_dict['chr'])+'\t'+str(node_dict['start'])+'\t'+str(
            node_dict['end'])+'\t'+node_name+'\t'+str(diff_vector[i][0])+'\n')
    out.close()


class DiscoRandomWalks:
    def __init__(self):
        pass

    def compute_reproducibility(self, m1_csr, m2_csr, transition=True, tmax=3, tmin=3):
        m1up = m1_csr
        m1down = m1up.transpose()
        m1down.setdiag(0)
        m1 = m1up+m1down

        m2up = m2_csr
        m2down = m2up.transpose()
        m2down.setdiag(0)
        m2 = m2up+m2down

        if transition:
            m1 = to_transition(m1)
            m2 = to_transition(m2)

        rowsums_1 = m1.sum(axis=1)
        nonzero_1 = [i for i in range(
            rowsums_1.shape[0]) if rowsums_1[i] > 0.0]
        rowsums_2 = m2.sum(axis=1)
        nonzero_2 = [i for i in range(
            rowsums_2.shape[0]) if rowsums_2[i] > 0.0]
        nonzero_total = len(list(set(nonzero_1).union(set(nonzero_2))))
        nonzero_total = 0.5*(1.0*len(list(set(nonzero_1))) +
                             1.0*len(list(set(nonzero_2))))

        nonzero_total += 1e-10

        scores = []
        if True:
            diff_vector = np.zeros((m1.shape[0], 1))
            for t in range(1, tmax+1):
                if t == 1:
                    rw1 = copy.deepcopy(m1)
                    rw2 = copy.deepcopy(m2)
                else:
                    rw1 = rw1.dot(m1)
                    rw2 = rw2.dot(m2)
                if t >= tmin:
                    diff_vector += abs(rw1-rw2).sum(axis=1)
                    diff = abs(rw1-rw2).sum()
                    scores.append(1.0*float(diff)/float(nonzero_total))

        ts = range(tmin, tmax+1)
        denom = len(ts)-1
        if tmin == tmax:
            auc = scores[0]
        else:
            auc = metrics.auc(range(len(ts)), scores)/denom
        reproducibility = 1.0-auc

        return reproducibility
