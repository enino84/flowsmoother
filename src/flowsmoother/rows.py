"""Row regressions of the modified Cholesky factor.

For row ``i`` with predecessors ``P(i)`` and centred anomalies ``X`` (N members)::

    beta_i  = argmin |y_i - X_P beta|^2 + a_i |beta - beta_i^c|^2
    a_i     = alpha * tr(X_P^T X_P) / p_i                (relative ridge)
    sigma_i^2 = |y_i - X_P beta_i|^2 / (N - 1)
    L[i, P(i)] = -beta_i,  L[i, i] = 1,  D[i, i] = 1 / sigma_i^2

``beta^c = 0`` is the classical EnKF-MC; ``beta^c = beta^clim`` is the climate
prior. Rows are solved in batches; predecessor lists are padded with the index
of an all-zero column, which leaves every result unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def pad_anomalies(X):
    """``(N, D)`` anomalies -> ``(D + 1, N)`` C-ordered, last row zero (the pad)."""
    X = np.asarray(X)
    out = np.zeros((X.shape[1] + 1, X.shape[0]), dtype=X.dtype)
    out[:-1] = X.T
    return out


SIGMA_MODES = ("insample", "dof", "loo")


def fit_rows(XT, targets, pred, plen, alpha, centre=None, form="auto",
             budget=2.0e7, sigma_floor=1e-8, sigma="insample"):
    """Ridge regressions for a batch of rows.

    XT      (D + 1, N) from :func:`pad_anomalies`.
    targets (R,) row indices; pred (R, P) predecessor indices padded with D.
    plen    (R,) true number of predecessors; centre (R, P) or None.
    sigma   how the residual variance is estimated:
            ``insample``  |y - X beta|^2 / (N - 1), the classical choice; with
                          more predecessors than members it is optimistic
            ``dof``       same, divided by N - 1 - (effective degrees of freedom)
            ``loo``       leave-one-member-out residuals, exact for the ridge:
                          r_k / (1 - H_kk); stays honest when p > N
    Returns ``beta (R, P)`` (zero on the pad) and ``sigma2 (R,)``.
    """
    if sigma not in SIGMA_MODES:
        raise ValueError(f"sigma must be one of {SIGMA_MODES}")
    D1, N = XT.shape
    R, P = pred.shape
    beta = np.zeros((R, P))
    sigma2 = np.empty(R)
    if form == "auto":
        form = "dual" if N <= P else "primal"
    per_row = P * N + (N * N if form == "dual" else P * P)
    step = max(1, int(budget // max(per_row, 1)))
    eye = np.eye(N if form == "dual" else P)
    pcount = np.maximum(plen, 1).astype(float)
    for s in range(0, R, step):
        sl = slice(s, min(R, s + step))
        Xg = np.asarray(XT[pred[sl]], dtype=float)            # (r, P, N)
        y = np.asarray(XT[targets[sl]], dtype=float)          # (r, N)
        tr = np.einsum("rpn,rpn->r", Xg, Xg)
        a = alpha * tr / pcount[sl]
        a = np.where(a > 0, a, 1.0)
        c0 = None if centre is None else centre[sl]
        if form == "dual":
            r0 = y if c0 is None else y - np.einsum("rp,rpn->rn", c0, Xg)
            K = np.einsum("rpn,rpm->rnm", Xg, Xg, optimize=True)
            lam, Q = np.linalg.eigh(K)
            lam = np.maximum(lam, 0.0)
            proj = np.einsum("rnk,rn->rk", Q, r0) / (lam + a[:, None])
            coef = np.einsum("rnk,rk->rn", Q, proj)
            b = np.einsum("rpn,rn->rp", Xg, coef)
            if c0 is not None:
                b += c0
            res = a[:, None] * coef
            df = (lam / (lam + a[:, None])).sum(axis=1)
            if sigma == "loo":                        # r_k / (1 - H_kk) = coef_k / [(K + aI)^-1]_kk
                res = coef / np.einsum("rnk,rnk,rk->rn", Q, Q, 1.0 / (lam + a[:, None]))
        else:
            G = np.einsum("rpn,rqn->rpq", Xg, Xg, optimize=True)
            rhs = np.einsum("rpn,rn->rp", Xg, y)
            if c0 is not None:
                rhs += a[:, None] * c0
            A = G + a[:, None, None] * eye
            b = np.linalg.solve(A, rhs[..., None])[..., 0]
            res = y - np.einsum("rp,rpn->rn", b, Xg)
            df = np.einsum("rpp->r", np.linalg.solve(A, G)) if sigma == "dof" else 0.0
            if sigma == "loo":
                hkk = np.einsum("rpn,rpn->rn", Xg, np.linalg.solve(A, Xg))
                res = res / np.maximum(1.0 - hkk, 1e-12)
        valid = pred[sl] < D1 - 1
        beta[sl] = np.where(valid, b, 0.0)
        denom = (N - 1.0 - df) if sigma == "dof" else (N - 1.0)
        denom = np.maximum(denom, 1.0)
        s2 = (res ** 2).sum(axis=1) / denom
        sigma2[sl] = np.maximum(s2, sigma_floor * np.maximum((y ** 2).sum(axis=1) / (N - 1.0), 1e-300))
    return beta, sigma2


@dataclass
class Factor:
    """Unit lower-triangular ``L`` (CSR) and the diagonal ``d`` of ``D``."""
    L: sp.csr_matrix
    d: np.ndarray

    @property
    def dim(self):
        return self.d.size

    def precision(self):
        return (self.L.T @ sp.diags(self.d) @ self.L).tocsc()

    def mean_predecessors(self):
        return (self.L.nnz - self.dim) / self.dim


def assemble(dim, groups):
    """Build the factor from ``(targets, pred, beta, sigma2)`` groups.

    Every row of the state must appear in exactly one group; predecessors must
    be lower indices (checked), so ``L`` is unit lower triangular.
    """
    rows, cols, vals = [np.arange(dim)], [np.arange(dim)], [np.ones(dim)]
    d = np.full(dim, np.nan)
    for targets, pred, beta, sigma2 in groups:
        valid = pred < dim
        if np.any(pred[valid] >= np.repeat(targets, valid.sum(axis=1))):
            raise ValueError("a predecessor is not a lower index")
        rows.append(np.repeat(targets, valid.sum(axis=1)))
        cols.append(pred[valid])
        vals.append(-beta[valid])
        d[targets] = 1.0 / sigma2
    if np.isnan(d).any():
        raise ValueError("some rows of the state have no regression")
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dim, dim))
    return Factor(L=L, d=d)
