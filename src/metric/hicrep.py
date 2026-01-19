#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2019 dejunlin <dejun.lin@gmail.com>
# Usage: hicrep.py
# Description: Compute HiCRep reproducibility stratum-corrected correlation score (SCCS).
# Reference: Genome Res. 2017 Nov;27(11):1939-1949. doi: 10.1101/gr.220640.117
# The algorithm first normalizes the input contact matrices by the total
# number of contacts and then for each chromosome: 1) mean-filter the input
# matrices with an input window size; 2) exclude common zero entries in
# the input matrices; 3) compute the SCC score. It doesn't have the
# procedure to bootstrap the window-size parameter
#
# Distributed under terms of the GNU General Public License v3.0.
import numpy as np
import scipy.sparse as sp
from typing import Union
from contextlib import suppress
import numpy as np
import cooler
import scipy.sparse as sp

def sccByDiag(m1: sp.coo_matrix, m2: sp.coo_matrix, nDiags: int):
    """Compute diagonal-wise hicrep SCC score for the two input matrices up to
    nDiags diagonals


    Args:
        m1 (sp.coo_matrix): input contact matrix 1
        m2 (sp.coo_matrix): input contact matrix 2
        nDiags (int): compute SCC scores for diagonals whose index is in the
        range of [1, nDiags)
    Returns: `float` hicrep SCC scores
    """
    # convert each diagonal to one row of a csr_matrix in order to compute
    # diagonal-wise correlation between m1 and m2
    m1D = upperDiagCsr(m1, nDiags)
    m2D = upperDiagCsr(m2, nDiags)
    nSamplesD = (m1D + m2D).getnnz(axis=1)
    rowSumM1D = m1D.sum(axis=1).A1
    rowSumM2D = m2D.sum(axis=1).A1
    # ignore zero-division warnings because the corresponding elements in the
    # output don't contribute to the SCC scores
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = m1D.multiply(m2D).sum(axis=1).A1 - rowSumM1D * rowSumM2D / nSamplesD
        rhoD = cov / np.sqrt(
            (m1D.power(2).sum(axis=1).A1 - np.square(rowSumM1D) / nSamplesD ) *
            (m2D.power(2).sum(axis=1).A1 - np.square(rowSumM2D) / nSamplesD ))
        wsD = nSamplesD * varVstran(nSamplesD)
        # Convert NaN and Inf resulting from div by 0 to zeros.
        # posinf and neginf added to fix behavior seen in 4DN datasets
        # 4DNFIOQLTI9G and DNFIH7MQHOR at 5kb where inf would be reported
        # as an SCC score
        wsNan2Zero = np.nan_to_num(wsD, copy=True, posinf=0.0, neginf=0.0)
        rhoNan2Zero = np.nan_to_num(rhoD, copy=True, posinf=0.0, neginf=0.0)

    return rhoNan2Zero @ wsNan2Zero / wsNan2Zero.sum()


def hicrepSCC(mat1: np.ndarray, mat2: np.ndarray, h: int = 0):
    mat1 = sp.coo_matrix(mat1)
    mat2 = sp.coo_matrix(mat2)
    nDiags = mat1.shape[0]

    m1 = trimDiags(mat1, nDiags, False)
    m2 = trimDiags(mat2, nDiags, False)
    del mat1
    del mat2

    # m1 = m1.astype(float) / n1
    # m2 = m2.astype(float) / n2
    if h > 0:
        m1 = meanFilterSparse(m1, h)
        m2 = meanFilterSparse(m2, h)
    scc = sccByDiag(m1, m2, nDiags)
    return scc

def trimDiags(a: sp.coo_matrix, iDiagMax: int, bKeepMain: bool):
    """Remove diagonal elements whose diagonal index is >= iDiagMax
    or is == 0

    Args:
        a: Input scipy coo_matrix
        iDiagMax: Diagonal offset cutoff
        bKeepMain: If true, keep the elements in the main diagonal;
        otherwise remove them

    Returns:
        coo_matrix with the specified diagonals removed
    """
    gDist = np.abs(a.row - a.col)
    idx = np.where((gDist < iDiagMax) & (bKeepMain | (gDist != 0)))
    return sp.coo_matrix((a.data[idx], (a.row[idx], a.col[idx])),
                         shape=a.shape, dtype=a.dtype)

def upperDiagCsr(m: sp.coo_matrix, nDiags: int):
    """Convert an input sp.coo_matrix into a sp.csr_matrix where each row in the
    the output corresponds to one diagonal of the upper triangle of the input.

    Args:
        m (sp.coo_matrix): input matrix
        nDiags (int): output diagonals with index in the range [1, nDiags)
        as rows of the output matrix
    Returns: `sp.csr_matrix` whose rows are the diagonals of the input
    """
    row = m.col - m.row
    idx = np.where((row > 0) & (row < nDiags))
    idxRowp1 = row[idx]
    # the diagonal index becomes the row index
    idxRow = idxRowp1 - 1
    # offset in the original diagonal becomes the column index
    idxCol = m.col[idx] - idxRowp1
    ans = sp.csr_matrix((m.data[idx], (idxRow, idxCol)),
                        shape=(nDiags - 1, m.shape[1]), dtype=m.dtype)
    ans.eliminate_zeros()
    return ans

def meanFilterSparse(a: sp.coo_matrix, h: int):
    """Apply a mean filter to an input sparse matrix. This convolves
    the input with a kernel of size 2*h + 1 with constant entries and
    subsequently reshape the output to be of the same shape as input

    Args:
        a: `sp.coo_matrix`, Input matrix to be filtered
        h: `int` half-size of the filter

    Returns:
        `sp.coo_matrix` filterd matrix
    """
    assert h > 0, "meanFilterSparse half-size must be greater than 0"
    assert sp.issparse(a) and a.getformat() == 'coo',\
        "meanFilterSparse input matrix is not scipy.sparse.coo_matrix"
    assert a.shape[0] == a.shape[1],\
        "meanFilterSparse cannot handle non-square matrix"
    fSize = 2 * h + 1
    # filter is a square matrix of constant 1 of shape (fSize, fSize)
    shapeOut = np.array(a.shape) + fSize - 1
    mToeplitz = sp.diags(np.ones(fSize),
                         np.arange(-fSize+1, 1),
                         shape=(shapeOut[1], a.shape[1]),
                         format='csr')
    ans = sp.coo_matrix((mToeplitz @ a) @ mToeplitz.T)
    # remove the edges since we don't care about them if we are smoothing
    # the matrix itself
    ansNoEdge = ans.tocsr()[h:(h+a.shape[0]), h:(h+a.shape[1])].tocoo()
    # Assign different number of neighbors to the edge to better
    # match what the original R implementation of HiCRep does
    rowDist2Edge = np.minimum(ansNoEdge.row, ansNoEdge.shape[0] - 1 - ansNoEdge.row)
    nDim1 = h + 1 + np.minimum(rowDist2Edge, h)
    colDist2Edge = np.minimum(ansNoEdge.col, ansNoEdge.shape[1] - 1 - ansNoEdge.col)
    nDim2 = h + 1 + np.minimum(colDist2Edge, h)
    nNeighbors = nDim1 * nDim2
    ansNoEdge.data /= nNeighbors
    return ansNoEdge

def varVstran(n: Union[int, np.ndarray]):
    """
    Calculate the variance of variance-stabilizing transformed
    (or `vstran()` in the original R implementation) data. The `vstran()` turns
    the input data into ranks, whose variance is only a function of the input
    size:
        ```
        var(1/n, 2/n, ..., n/n) = (1 - 1/(n^2))/12
        ```
    or with Bessel's correction:
        ```
        var(1/n, 2/n, ..., n/n, ddof=1) = (1 + 1.0/n)/12
        ```
    See section "Variance stabilized weights" in reference for more detail:
    https://genome.cshlp.org/content/early/2017/10/06/gr.220640.117

    Args:
        n (Union(int, np.ndarray)): size of the input data
    Returns: `Union(int, np.ndarray)` variance of the ranked input data with Bessel's
    correction
    """
    with suppress(ZeroDivisionError), np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n < 2, np.nan, (1 + 1.0 / n) / 12.0)