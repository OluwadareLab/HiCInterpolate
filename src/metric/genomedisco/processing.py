import sys
import numpy as np
import gzip
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def read_nodes_from_bed(bedfile, blacklistfile='NA'):

    blacklist = {}
    if blacklistfile != 'NA':
        for line in gzip.open(blacklistfile):
            items = line.strip().split('\t')
            chromo, start, end = items[0], int(items[1]), int(items[2])
            if chromo not in blacklist:
                blacklist[chromo] = []
            blacklist[chromo].append((start, end))

    nodes = {}
    nodes_idx = {}
    node_c = 0
    blacklisted_nodes = []
    for line in gzip.open(bedfile, 'r'):
        items = line.strip().split('\t')
        chromo = items[0]
        start = int(items[1])
        end = int(items[2])

        node = items[3]
        if len(items) > 4:
            include = items[4]

        if node in nodes.keys():
            sys.exit()
        if node not in nodes.keys():
            nodes[node] = {}
            nodes[node]['idx'] = node_c
            nodes[node]['chr'] = chromo
            nodes[node]['start'] = start
            nodes[node]['end'] = end
            if len(items) > 4:
                nodes[node]['include'] = include
            nodes_idx[node_c] = node

            if chromo in blacklist:
                for blacklist_item in blacklist[chromo]:
                    if (start <= blacklist_item[0] and end >= blacklist_item[0]) or (start <= blacklist_item[1] and end >= blacklist_item[1]) or (start >= blacklist_item[0] and end <= blacklist_item[1]):
                        blacklisted_nodes.append(node_c)

            node_c += 1

    return nodes, nodes_idx, blacklisted_nodes


def filter_nodes(m, to_remove):

    if len(to_remove) == 0:
        return m

    nonzeros = m.nonzero()

    r_idx = [i for i, x in enumerate(nonzeros[0]) if x not in to_remove]
    c_idx = [i for i, x in enumerate(nonzeros[1]) if x not in to_remove]
    keep = list(set(r_idx).union(set(c_idx)))

    coo_mat = m.tocoo()

    return csr_matrix((coo_mat.data[keep], (coo_mat.row[keep], coo_mat.col[keep])), shape=m.get_shape(), dtype=float)


def construct_csr_matrix_from_data_and_nodes(matrix, blacklisted_nodes=[], remove_diag=True):

    csr_m = csr_matrix(matrix, dtype=float)
    if remove_diag:
        csr_m.setdiag(0)
    return filter_nodes(csr_m, blacklisted_nodes)


def write_matrix_from_csr_and_nodes(csr_m, nodes_idx, outname):

    coo_m = coo_matrix(csr_m)
    i = coo_m.row
    j = coo_m.col
    v = coo_m.data

    out = gzip.open(outname, 'w')

    for idx in range(len(i)):
        n1, n2, val = nodes_idx[i[idx]], nodes_idx[j[idx]], v[idx]
        out.write('\t'.join([str(n1), str(n2), str(val)])+'\n')
    out.close()


def old_construct_csr_matrix_from_data_and_nodes(f, nodes, blacklisted_nodes, remove_diag=True):

    total_nodes = len(nodes.keys())
    mdata = np.loadtxt(f)

    i = map(lambda x: nodes[str(int(x))]['idx'], mdata[:, 0])
    j = map(lambda x: nodes[str(int(x))]['idx'], mdata[:, 1])

    ij = np.array([i, j])
    mini = ij.min(axis=0)
    maxi = ij.max(axis=0)
    mini_maxi_ij = np.array([mini, maxi]).T

    rows = [tuple(row) for row in mini_maxi_ij]

    if len(rows) > len(set(rows)):
        print("=============== Warning: Your file contains duplicate interactions! Please ensure that each interaction is listed once, then re-run. In the meantime, we will run this analysis using the sum of all counts encountered per interaction")

    csr_m = csr_matrix((mdata[:, 2], (mini_maxi_ij[:, 0], mini_maxi_ij[:, 1])), shape=(
        total_nodes, total_nodes), dtype=float)
    if remove_diag:
        csr_m.setdiag(0)

    return filter_nodes(csr_m, blacklisted_nodes)
