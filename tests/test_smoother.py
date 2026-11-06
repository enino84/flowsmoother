"""The joint estimator: structures, priors and caches."""
import numpy as np

from flowsmoother.clim import shift_same_time_centre
from flowsmoother.joint import build_joint_graph
from flowsmoother.metrics import rmse, spread
from flowsmoother.smoother import JointEstimator
from flowsmoother.wake import MeanWind, wakes_for

A = 6.371e6


def setup(band, N=12, M=60, seed=0):
    rng = np.random.default_rng(seed)
    lag = 24 * 3600.0
    lat = np.radians(band.lat_deg)[:, None]
    u = np.broadcast_to(70.0 * np.cos(lat), (band.nlat, band.nlon)).copy()
    v = np.zeros_like(u)
    wakes = wakes_for(["wake", "patch", "wake50"], band, MeanWind([0.0, lag], [u, u], [v, v], A),
                      lag, lag, width=1)
    clim = rng.standard_normal((M, 2 * band.dim)).astype(np.float32)
    est = JointEstimator(band, clim, alpha_clim=0.1)
    Za, Zb = rng.standard_normal((N, band.dim)), rng.standard_normal((N, band.dim))
    est.set_ensemble(Za, Zb)
    est.set_wake(wakes)
    return est, wakes


def test_structures_differ_only_in_the_h_rows_of_xb(band):
    est, wakes = setup(band)
    g = {s: est.graph(s, 1) for s in ("square", "wake", "patch", "wake50")}
    for name in ("a_u", "a_v", "a_h", "b_u", "b_v"):
        for s in ("wake", "patch", "wake50"):
            assert np.array_equal(g[s].groups[name].pred, g["square"].groups[name].pred)
    assert not np.array_equal(g["wake"].groups["b_h"].plen, g["square"].groups["b_h"].plen)
    # the size control has, row by row, as many predecessors as the wake
    assert np.array_equal(g["patch"].groups["b_h"].plen, g["wake"].groups["b_h"].plen)
    assert not np.array_equal(g["patch"].groups["b_h"].pred, g["wake"].groups["b_h"].pred)
    assert wakes["wake50"].length.mean() < wakes["wake"].length.mean()


def test_cached_groups_equal_a_fresh_fit(band):
    est, wakes = setup(band)
    first = est.factor("square", 1, 0.5)
    second = est.factor("wake", 1, 0.5)                 # reuses the a and b_uv groups
    fresh, _ = setup(band)
    alone = fresh.factor("wake", 1, 0.5)
    assert abs(second.L - alone.L).max() == 0.0
    assert np.array_equal(second.d, alone.d)
    m, n = band.m, band.dim
    rows = np.r_[0:n, n:n + 2 * m]                      # everything except h of x_b
    assert abs(first.L[rows] - second.L[rows]).max() == 0.0


def test_priors(band):
    est, _ = setup(band)
    off = est.factor("wake", 1, 0.5, "off")
    full_big = est.factor("wake", 1, 1e9, "full")
    cen = est.centres("wake", 1, "full")
    g = est.graph("wake", 1)
    grp = g.groups["b_h"]
    k = 40
    row = full_big.L[grp.targets[k]].toarray().ravel()
    assert np.allclose(-row[grp.pred[k, :grp.plen[k]]], cen["b_h"][k, :grp.plen[k]], atol=1e-6)
    assert abs(off.L - full_big.L).max() > 1e-3
    same = est.centres("wake", 1, "same")
    n_cross = grp.plen - grp.n_same
    assert np.all(same["b_h"][k, :n_cross[k]] == 0.0)                  # no climate across time
    a = g.groups["a_h"]
    assert np.array_equal(same["b_h"][k, n_cross[k]:grp.plen[k]], cen["a_h"][k, :a.plen[k]])


def test_shift_same_time_centre_alignment(band):
    g = build_joint_graph(band, 1, cross_h=band.square(2))
    a, b = g.groups["a_h"], g.groups["b_h"]
    beta_a = np.where(a.pred < g.dim, a.pred + 0.5, 0.0)               # value = index + 0.5
    out = shift_same_time_centre(beta_a, a, b)
    for k in (0, 31, band.m - 1):
        p = b.pred[k, :b.plen[k]]
        same = p >= band.dim
        assert np.all(out[k, :b.plen[k]][~same] == 0.0)
        assert np.array_equal(out[k, :b.plen[k]][same], p[same] - band.dim + 0.5)


def test_filter_factor_and_scores(band):
    est, _ = setup(band)
    fac = est.filter_factor(1, 0.5)
    assert fac.dim == band.dim
    assert est.filter_factor(1, 0.5, "same").dim == band.dim
    rng = np.random.default_rng(1)
    z = rng.standard_normal(band.dim)
    r = rmse(band, z, np.zeros(band.dim))
    assert r.shape == (4,) and abs(r[3] ** 2 - np.mean(r[:3] ** 2)) < 1e-12
    Z = rng.standard_normal((500, band.dim)) * 2.0
    assert np.allclose(spread(band, Z), 2.0, atol=0.05)
