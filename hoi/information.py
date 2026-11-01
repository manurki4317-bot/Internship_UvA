"""O-información y S-información gaussianas, vectorizadas.

Mismas fórmulas que soinfo_from_covmat / high_order de HOI_connectivity.py, pero calculando
todos los n-pletes a la vez en lugar de un bucle con determinantes.
"""
import numpy as np
from itertools import combinations
from scipy.special import digamma
from scipy.stats import norm


def gaussian_copula_cov(data):
    """data: (T, N). Devuelve la covarianza de la copula gaussiana (N, N)."""
    T = len(data)
    ranks = np.argsort(np.argsort(data, axis=0), axis=0) + 1
    g = norm.ppf(ranks / (T + 1))
    g[~np.isfinite(g)] = 0
    return g.T @ g / (T - 1)


def _bias(N, T):
    n = np.arange(1, N + 1)
    return 0.5 * (N * np.log(2 / (T - 1)) + np.sum(digamma((T - n) / 2)))


def _ent(x, det):
    with np.errstate(divide="ignore", invalid="ignore"):
        det = np.where(det == 0, np.nan, np.abs(det))
        return 0.5 * np.log(((2 * np.pi * np.e) ** x) * det)


def soinfo_from_covs(C, T):
    """O-info y S-info de una pila de covarianzas C con forma (M, n, n) (M sistemas de n variables)."""
    n = C.shape[1]
    det = np.linalg.det(C)
    var = np.diagonal(C, axis1=1, axis2=2)
    bN, bN1, b1 = _bias(n, T), _bias(n - 1, T), _bias(1, T)
    hN = _ent(n, det) - bN
    tc = np.sum(_ent(1, var) - b1, axis=1) - hN
    hred = 0
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        sub = C[:, idx][:, :, idx]
        hred = hred + _ent(n - 1, np.linalg.det(sub)) - bN1
    dtc = hred - (n - 1) * hN
    return tc - dtc, tc + dtc


def soinfo_triplets(cov, T, n=3):
    """Devuelve (nplets, oinfo, sinfo) para todas las combinaciones de `n` variables."""
    nplets = np.array(list(combinations(range(len(cov)), n)))
    C = cov[nplets[:, :, None], nplets[:, None, :]]  # (M, n, n)
    oinfo, sinfo = soinfo_from_covs(C, T)
    return nplets, oinfo, sinfo
