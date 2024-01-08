import warnings
import numpy as np
from ot.bregman import sinkhorn, sinkhorn2
from ot.utils import list_to_array
import math
import sys
from backend import get_backend
import torch

def get_E_F(N, M, backend):
    nx = backend
    mid_para = nx.sqrt((1/(N**2) + 1/(M**2)))

    a_n = nx.arange(start=1, stop=N+1)
    b_m = nx.arange(start=1, stop=M+1)
    row_col_matrix = nx.meshgrid(a_n, b_m)  #indexing xy
    row = row_col_matrix[0].T / N # row = (i+1)/N
    col = row_col_matrix[1].T / M # col = (j+1)/M

    l = nx.abs(row - col) / mid_para

    E =  1 / ((row - col) ** 2 + 1)
    F = l**2
    return E, F

def opw_sinkhorn(a, b, M,lambda1=50, lambda2=0.1, delta=1, method='sinkhorn', numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    """
    Solve the entropic regularization OPW and return the OT matrix

    Args:
        X (ndarray): view1
        Y (ndarray): view2
        lambda1 (int, optional): weight of first term. Defaults to 50.
        lambda2 (float, optional): weight of second term. Defaults to 0.1.
        delta (int, optional): _description_. Defaults to 1.

    Returns:
        ot_plan:  ot_plan is the transport plan
    """
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    reg = lambda2

    E, F = get_E_F(a.shape[0], b.shape[0], backend=nx)
    M_hat = M - lambda1 * E + lambda2 * (F / (2 * delta ** 2) + nx.log(delta * nx.sqrt(2 * math.pi)))

    return sinkhorn(a, b, M_hat, reg, method=method, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, warn=warn, **kwargs)

def opw_sinkhorn2(a, b, M,lambda1=50, lambda2=0.1, delta=1, method='sinkhorn', numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    r"""
    Solve the entropic regularization OPW and return the loss

    Args:
        X (ndarray): view1
        Y (ndarray): view2
        lambda1 (int, optional): weight of first term. Defaults to 50.
        lambda2 (float, optional): weight of second term. Defaults to 0.1.
        delta (int, optional): _description_. Defaults to 1.

    Returns:
        distance: distance is the distance between views
    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    reg = lambda2

    E, F = get_E_F(a.shape[0], b.shape[0], backend=nx)
    M_hat = M - lambda1 * E + lambda2 * (F / (2 * delta ** 2) + nx.log(delta * nx.sqrt(2 * math.pi)))

    return sinkhorn2(a, b, M_hat, reg,method=method, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, warn=warn, **kwargs)

def opw_distance(M):
    rows, cols = M.shape
    a, b = np.ones((rows,)) / rows, np.ones((cols,)) / cols
    lambd = 1e-1
    dist = opw_sinkhorn2(a, b, M, lambd, numItermax=20)
    return torch.tensor(dist)