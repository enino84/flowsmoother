"""Analyses against dense references."""
import numpy as np

from flowsmoother.analysis import (TriangularSolver, enks_mc, gaspari_cohn, letks,
                                   solve_state_space)
from flowsmoother.joint import build_joint_graph, fit_graph
from flowsmoother.rows import pad_anomalies


def joint_ensemble(band, N, rng):
    A = rng.standard_normal((2 * band.dim, 2 * band.dim)) / np.sqrt(2 * band.dim)
    return rng.standard_normal((N, 2 * band.dim)) @ (np.eye(2 * band.dim) + A).T


def test_triangular_solver(band):
    rng = np.random.default_rng(0)
    graph = build_joint_graph(band, 1, cross_h=band.square(1))
    Z = rng.standard_normal((30, graph.dim))
    fac = fit_graph(pad_anomalies(Z - Z.mean(axis=0)), graph, 0.5)
    ts = TriangularSolver(fac.L)
    b = rng.standard_normal((graph.dim, 3))
    assert np.allclose(fac.L @ ts.solve(b), b, atol=1e-10)
    assert np.allclose(fac.L.T @ ts.solve_t(b), b, atol=1e-10)


def test_observation_space_equals_state_space(band):
    rng = np.random.default_rng(1)
    graph = build_joint_graph(band, 1, cross_h=band.square(2))
    Z = joint_ensemble(band, 25, rng)
    fac = fit_graph(pad_anomalies(Z - Z.mean(axis=0)), graph, 0.5)
    cells = np.sort(rng.choice(band.m, 40, replace=False))
    obs = band.dim + 2 * band.m + cells                     # h at time b
    y = rng.standard_normal(40)
    Za = enks_mc(Z, fac, obs, y, 0.3, np.random.default_rng(7))
    E = np.random.default_rng(7).standard_normal((40, 25)) * np.sqrt(0.3)
    Zs = solve_state_space(Z, fac.precision(), obs, y, 0.3, E)
    assert np.allclose(Za, Zs, atol=1e-8)


def test_full_graph_reproduces_the_dense_smoother(tiny):
    """All predecessors, N > dim, no ridge: B^-1 is the inverse sample covariance."""
    rng = np.random.default_rng(2)
    r = 2                                                   # covers the 3 x 5 band
    allc = np.tile(np.arange(tiny.m), (tiny.m, 1))
    graph = build_joint_graph(tiny, r, cross_h=allc, cross_uv=allc)
    assert graph.mean_predecessors() == (graph.dim - 1) / 2.0     # full lower triangle
    N = 400
    Z = joint_ensemble(tiny, N, rng)
    X = Z - Z.mean(axis=0)
    fac = fit_graph(pad_anomalies(X), graph, 1e-14)
    P = X.T @ X / (N - 1)
    assert np.allclose(np.linalg.inv(fac.precision().toarray()), P, atol=1e-7)
    cells = np.array([1, 4, 7, 13])
    obs = tiny.dim + 2 * tiny.m + cells
    y = rng.standard_normal(4)
    Za = enks_mc(Z, fac, obs, y, 0.5, np.random.default_rng(5))
    E = np.random.default_rng(5).standard_normal((4, N)) * np.sqrt(0.5)
    H = np.zeros((4, graph.dim))
    H[np.arange(4), obs] = 1
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + 0.5 * np.eye(4))
    dense = Z + (K @ (y[:, None] + E - (Z @ H.T).T)).T
    assert np.allclose(Za, dense, atol=1e-7)
    assert not np.allclose(Za[:, :tiny.dim], Z[:, :tiny.dim])    # the past is updated


def test_without_cross_time_links_the_past_does_not_move(band):
    rng = np.random.default_rng(3)
    none = np.full((band.m, 1), -1)
    graph = build_joint_graph(band, 1, cross_h=None, cross_uv=none)
    Z = joint_ensemble(band, 20, rng)
    fac = fit_graph(pad_anomalies(Z - Z.mean(axis=0)), graph, 0.5)
    cells = np.sort(rng.choice(band.m, 30, replace=False))
    Za = enks_mc(Z, fac, band.dim + 2 * band.m + cells, rng.standard_normal(30), 0.2,
                 np.random.default_rng(1))
    assert np.array_equal(Za[:, :band.dim], Z[:, :band.dim])
    assert not np.allclose(Za[:, band.dim:], Z[:, band.dim:])


def test_gaspari_cohn():
    z = np.linspace(0, 2.5, 200)
    g = gaspari_cohn(z)
    assert g[0] == 1.0 and np.all(g[z >= 2] == 0) and np.all(np.diff(g) <= 1e-12)
    assert abs(gaspari_cohn(np.array([1.0]))[0] - 5.0 / 24.0) < 1e-12


def test_letks_with_all_observations_is_the_global_etkf(tiny):
    rng = np.random.default_rng(4)
    N, m = 12, tiny.m
    Z = joint_ensemble(tiny, N, rng).reshape(N, 2, tiny.dim).transpose(1, 0, 2)
    cells = np.array([0, 3, 6, 9, 14])
    y = rng.standard_normal(5)
    r = 0.4
    out = letks(Z, tiny, cells, y, r, radius=2, taper="box")
    hb = Z[1][:, 2 * m + cells]
    Y = (hb - hb.mean(axis=0)).T                            # (p, N)
    Pa = np.linalg.inv(Y.T @ Y / r + (N - 1) * np.eye(N))
    wbar = Pa @ Y.T @ (y - hb.mean(axis=0)) / r
    lam, Q = np.linalg.eigh((N - 1) * Pa)
    W = Q @ np.diag(np.sqrt(lam)) @ Q.T + wbar[:, None]
    for t in (0, 1):
        mean = Z[t].mean(axis=0)
        ref = mean + (W.T @ (Z[t] - mean))
        assert np.allclose(out[t], ref, atol=1e-9)


def test_letks_leaves_unobserved_regions_alone(band):
    rng = np.random.default_rng(5)
    N = 10
    Z = rng.standard_normal((2, N, band.dim))
    cells = np.array([2 * band.nc + 2])
    out = letks(Z, band, cells, np.array([0.3]), 0.1, radius=1, taper="box")
    far = np.where(band.chebyshev(np.full(band.m, cells[0]), np.arange(band.m)) > 1)[0]
    near = np.where(band.chebyshev(np.full(band.m, cells[0]), np.arange(band.m)) <= 1)[0]
    for f in range(3):
        assert np.allclose(out[:, :, f * band.m + far], Z[:, :, f * band.m + far], atol=1e-12)
        assert not np.allclose(out[:, :, f * band.m + near], Z[:, :, f * band.m + near])
