"""Row regressions: recovery of a known factor, forms, centre, padding."""
import numpy as np
import scipy.sparse.linalg as spla

from flowsmoother.joint import build_filter_graph, build_joint_graph, fit_graph
from flowsmoother.rows import assemble, fit_rows, pad_anomalies


def random_factor(graph, rng, scale=0.15):
    parts = []
    for g in graph.groups.values():
        beta = np.where(g.pred < graph.dim, scale * rng.standard_normal(g.pred.shape), 0.0)
        beta /= np.maximum(np.sqrt(g.plen), 1.0)[:, None]
        parts.append((g.targets, g.pred, beta, rng.uniform(0.5, 2.0, g.targets.size)))
    return assemble(graph.dim, parts), parts


def sample(fac, n, rng):
    eps = rng.standard_normal((fac.dim, n)) / np.sqrt(fac.d)[:, None]
    return spla.spsolve_triangular(fac.L.tocsr(), eps, lower=True).T


def test_recovers_a_known_factor(band):
    rng = np.random.default_rng(0)
    graph = build_joint_graph(band, 1, cross_h=band.square(1))
    true, parts = random_factor(graph, rng)
    Z = sample(true, 20000, rng)
    XT = pad_anomalies(Z - Z.mean(axis=0))
    fac = fit_graph(XT, graph, alpha=1e-10)
    assert abs(fac.L - true.L).max() < 0.06
    assert np.allclose(1.0 / fac.d, 1.0 / true.d, rtol=0.06)


def test_dual_and_primal_forms_agree(band):
    rng = np.random.default_rng(1)
    graph = build_filter_graph(band, 2)
    g = graph.groups["f_h"]
    for N in (15, 40):
        X = rng.standard_normal((N, graph.dim))
        XT = pad_anomalies(X - X.mean(axis=0))
        centre = np.where(g.pred < graph.dim, 0.1 * rng.standard_normal(g.pred.shape), 0.0)
        for c in (None, centre):
            bd, sd = fit_rows(XT, g.targets, g.pred, g.plen, 0.7, centre=c, form="dual")
            bp, sp_ = fit_rows(XT, g.targets, g.pred, g.plen, 0.7, centre=c, form="primal")
            assert np.allclose(bd, bp, atol=1e-9)
            assert np.allclose(sd, sp_, rtol=1e-8)


def test_padding_and_batching_do_not_change_the_result(band):
    rng = np.random.default_rng(2)
    graph = build_filter_graph(band, 1)
    g = graph.groups["f_v"]
    X = rng.standard_normal((12, graph.dim))
    XT = pad_anomalies(X - X.mean(axis=0))
    b0, s0 = fit_rows(XT, g.targets, g.pred, g.plen, 0.3)
    wide = np.concatenate([g.pred, np.full((g.pred.shape[0], 5), graph.dim)], axis=1)
    b1, s1 = fit_rows(XT, g.targets, wide, g.plen, 0.3, budget=500.0)
    assert np.allclose(b0, b1[:, :g.pred.shape[1]], atol=1e-12)
    assert np.all(b1[:, g.pred.shape[1]:] == 0.0)
    assert np.allclose(s0, s1, rtol=1e-12)


def test_zero_centre_is_the_classical_row(band):
    rng = np.random.default_rng(3)
    graph = build_filter_graph(band, 2)
    g = graph.groups["f_h"]
    X = rng.standard_normal((20, graph.dim))
    XT = pad_anomalies(X - X.mean(axis=0))
    b0, s0 = fit_rows(XT, g.targets, g.pred, g.plen, 1.0)
    b1, s1 = fit_rows(XT, g.targets, g.pred, g.plen, 1.0, centre=np.zeros(g.pred.shape))
    assert np.array_equal(b0, b1) and np.array_equal(s0, s1)


def test_limits_of_the_ridge(band):
    rng = np.random.default_rng(4)
    graph = build_filter_graph(band, 1)
    g = graph.groups["f_u"]                              # 4 predecessors, N = 60: OLS exists
    X = rng.standard_normal((60, graph.dim))
    XT = pad_anomalies(X - X.mean(axis=0))
    centre = np.where(g.pred < graph.dim, rng.standard_normal(g.pred.shape), 0.0)
    b_inf, _ = fit_rows(XT, g.targets, g.pred, g.plen, 1e9, centre=centre)
    assert np.allclose(b_inf, centre, atol=1e-6)         # all climate
    b_0, _ = fit_rows(XT, g.targets, g.pred, g.plen, 1e-12, centre=centre)
    k = 37
    P = g.pred[k, :g.plen[k]]
    Xc = (X - X.mean(axis=0))
    ols = np.linalg.lstsq(Xc[:, P], Xc[:, g.targets[k]], rcond=None)[0]
    assert np.allclose(b_0[k, :g.plen[k]], ols, atol=1e-8)   # all ensemble


def test_leave_one_out_variance_is_exact(band):
    """sigma='loo' equals refitting the row without each member in turn."""
    rng = np.random.default_rng(5)
    graph = build_filter_graph(band, 1)
    g = graph.groups["f_h"]
    N = 9
    X = rng.standard_normal((N, graph.dim))            # used as given (no re-centring)
    XT = pad_anomalies(X)
    k = 50
    P = g.pred[k, :g.plen[k]]
    y, Xp = X[:, g.targets[k]], X[:, P]
    a = 0.4 * np.trace(Xp.T @ Xp) / P.size
    loo = []
    for j in range(N):
        keep = np.arange(N) != j
        b = np.linalg.solve(Xp[keep].T @ Xp[keep] + a * np.eye(P.size), Xp[keep].T @ y[keep])
        loo.append(y[j] - Xp[j] @ b)
    expect = np.sum(np.square(loo)) / (N - 1)
    for form in ("dual", "primal"):
        _, s2 = fit_rows(XT, g.targets[k:k + 1], g.pred[k:k + 1], g.plen[k:k + 1], 0.4,
                         form=form, sigma="loo")
        assert s2[0] == np.float64(expect) or abs(s2[0] - expect) < 1e-10 * expect
    _, s_in = fit_rows(XT, g.targets, g.pred, g.plen, 0.4)
    _, s_loo = fit_rows(XT, g.targets, g.pred, g.plen, 0.4, sigma="loo")
    _, s_dof = fit_rows(XT, g.targets, g.pred, g.plen, 0.4, sigma="dof")
    assert np.all(s_loo > s_in) and np.all(s_dof > s_in)
