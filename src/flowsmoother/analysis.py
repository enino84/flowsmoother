"""Analyses: EnKS-MC (modified-Cholesky smoother) and LETKS (reference).

EnKS-MC. With ``B^{-1} = L^T D L`` the stochastic analysis

    (B^{-1} + H^T R^{-1} H) dZ = H^T R^{-1} (Y + E - H Z_b)

is solved in its algebraically identical observation-space form, which needs
only triangular solves with the sparse factor (no fill-in, no iteration)::

    V  = L^{-T} H^T                    (one solve per observation)
    S  = V^T D^{-1} V + R              (= H B H^T + R)
    dZ = L^{-1} D^{-1} V S^{-1} (Y + E - H Z_b)

``solve_state_space`` implements the first form directly and is used by the
tests to check that both give the same increment.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Band
from .rows import Factor


class TriangularSolver:
    """Solves with the unit lower-triangular ``L`` and with ``L^T``."""

    def __init__(self, L):
        self.dim = L.shape[0]
        self._lu = spla.splu(sp.csc_matrix(L), permc_spec="NATURAL",
                             diag_pivot_thresh=0.0,
                             options=dict(SymmetricMode=False, ColPerm="NATURAL"))
        if not np.array_equal(self._lu.perm_r, np.arange(self.dim)):
            raise RuntimeError("row pivoting happened in a triangular factor")

    def solve(self, b):          # L x = b
        return self._lu.solve(np.asarray(b, dtype=float))

    def solve_t(self, b):        # L^T x = b
        return self._lu.solve(np.asarray(b, dtype=float), trans="T")


def enks_mc(Zb, factor: Factor, obs_index, y, r_var, rng, solver=None, block=512):
    """Stochastic EnKS-MC analysis in observation space.

    Zb         (N, D) background members (normalized units).
    obs_index  (p,) indices of the observed components of the state.
    y, r_var   (p,) observations and their error variances (same units).
    Returns the analysis members ``(N, D)``.
    """
    N, D = Zb.shape
    obs_index = np.asarray(obs_index)
    p = obs_index.size
    r_var = np.broadcast_to(np.asarray(r_var, dtype=float), (p,))
    solver = solver or TriangularSolver(factor.L)
    dinv = 1.0 / factor.d
    V = np.empty((D, p))
    for s in range(0, p, block):
        e = min(p, s + block)
        Ht = np.zeros((D, e - s))
        Ht[obs_index[s:e], np.arange(e - s)] = 1.0
        V[:, s:e] = solver.solve_t(Ht)
    DV = V * dinv[:, None]
    S = V.T @ DV
    S[np.diag_indices(p)] += r_var
    E = rng.standard_normal((p, N)) * np.sqrt(r_var)[:, None]
    innov = y[:, None] + E - Zb[:, obs_index].T
    C = np.linalg.solve(S, innov)
    dZ = solver.solve(DV @ C)
    return Zb + dZ.T


def solve_state_space(Zb, precision, obs_index, y, r_var, E):
    """Reference: the state-space system with a sparse direct solve (tests)."""
    N, D = Zb.shape
    p = len(obs_index)
    H = sp.csr_matrix((np.ones(p), (np.arange(p), obs_index)), shape=(p, D))
    Rinv = sp.diags(1.0 / np.broadcast_to(np.asarray(r_var, float), (p,)))
    A = (precision + H.T @ Rinv @ H).tocsc()
    rhs = H.T @ (Rinv @ (y[:, None] + E - Zb[:, obs_index].T))
    return Zb + spla.splu(A).solve(np.asarray(rhs)).T


def gaspari_cohn(z):
    """Gaspari-Cohn fifth-order taper; support ``z < 2``."""
    z = np.abs(z)
    out = np.zeros_like(z, dtype=float)
    a = z <= 1
    b = (z > 1) & (z < 2)
    za, zb = z[a], z[b]
    out[a] = -0.25 * za**5 + 0.5 * za**4 + 0.625 * za**3 - 5.0 / 3.0 * za**2 + 1.0
    out[b] = zb**5 / 12.0 - 0.5 * zb**4 + 0.625 * zb**3 + 5.0 / 3.0 * zb**2 \
        - 5.0 * zb + 4.0 - 2.0 / (3.0 * zb)
    return out


def letks(Zb_blocks, band: Band, obs_cells, y, r_var, radius, taper="box",
          obs_block=-1, budget=3.0e7):
    """Local ensemble transform Kalman smoother (deterministic).

    The ETKF weights of every cell are computed from the observed block (the
    ``h`` field of the block ``obs_block``, by default the last one, i.e. time
    ``t``) and applied to the anomalies of every block, i.e. of both times.

    Zb_blocks  (T, N, 3m) members of each time block (normalized units).
    obs_cells  (p,) observed cells (``h`` observed); y, r_var (p,).
    radius     box radius in cells (``taper='box'``) or Gaspari-Cohn half width
               (``taper='gc'``, observations out to ``2 * radius``).
    """
    Zb_blocks = np.asarray(Zb_blocks, dtype=float)
    T, N, D = Zb_blocks.shape
    m = band.m
    r_var = np.broadcast_to(np.asarray(r_var, dtype=float), np.shape(y))
    reach = radius if taper == "box" else 2 * radius
    reach = min(reach, (band.nc - 1) // 2)
    nb = band.square(reach)                                   # (m, K) cells
    obs_id = np.full(m + 1, -1)
    obs_id[obs_cells] = np.arange(len(obs_cells))
    loc = obs_id[np.where(nb >= 0, nb, m)]                    # (m, K) obs ids, -1 none
    if taper == "gc":
        offs = np.array([(di, dj) for di in range(-reach, reach + 1)
                         for dj in range(-reach, reach + 1)], dtype=float)
        tap = gaspari_cohn(np.hypot(offs[:, 0], offs[:, 1]) / float(radius))
    else:
        tap = np.ones(nb.shape[1])
    keep_cols = (loc >= 0).any(axis=0)
    loc, tap = loc[:, keep_cols], tap[keep_cols]
    K = loc.shape[1]
    mean = Zb_blocks.mean(axis=1)                             # (T, D)
    anom = Zb_blocks - mean[:, None, :]                       # (T, N, D)
    hb = Zb_blocks[obs_block][:, 2 * m + np.asarray(obs_cells)]   # (N, p)
    hb_mean = hb.mean(axis=0)
    Yp = np.concatenate([(hb - hb_mean).T, np.zeros((1, N))])     # (p + 1, N), pad row
    dp = np.concatenate([y - hb_mean, [0.0]])
    wp = np.concatenate([1.0 / r_var, [0.0]])
    out = np.empty_like(Zb_blocks)
    eye = np.eye(N)
    step = max(1, int(budget // max(K * N, 1)))
    for s in range(0, m, step):
        cs = np.arange(s, min(m, s + step))
        ids = loc[cs]                                         # (c, K)
        Y = Yp[ids]                                           # (c, K, N)
        w = wp[ids] * tap[None, :] * (ids >= 0)               # (c, K)
        YtW = Y.transpose(0, 2, 1) * w[:, None, :]            # (c, N, K)
        lam, Q = np.linalg.eigh(YtW @ Y + (N - 1.0) * eye)
        Pa = (Q / lam[:, None, :]) @ Q.transpose(0, 2, 1)
        Wm = (Q * np.sqrt((N - 1.0) / lam)[:, None, :]) @ Q.transpose(0, 2, 1)
        wbar = np.einsum("cnm,cm->cn", Pa, np.einsum("cnk,ck->cn", YtW, dp[ids]))
        Wt = Wm + wbar[:, :, None]                            # column k: weights of member k
        for f in range(3):
            idx = f * m + cs
            Xl = anom[:, :, idx]                              # (T, N, c)
            out[:, :, idx] = mean[:, None, idx] + np.einsum("tnc,cnk->tkc", Xl, Wt)
    return out
