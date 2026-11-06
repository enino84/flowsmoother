"""Checks that need swesphere itself (a few seconds)."""
import numpy as np
import pytest

from flowsmoother.ensemble import draw_network, make_twin, observe_h
from flowsmoother.grid import Band
from flowsmoother.model import DAY, Propagator, make_model
from flowsmoother.wake import MeanWind, wake_cells

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def model():
    return make_model("two_jets")[0]


def test_no_sponge_and_band_rows(model):
    assert model.n_pole_rows == 0                       # swesphere 0.6.0 default: no sponge
    band = Band.from_model(model)
    assert (band.r0, band.r1) == (10, 55)
    assert band.lat_deg[band.r0] == pytest.approx(62.727, abs=1e-3)
    assert band.lat_deg[band.r1] == pytest.approx(-60.0, abs=1e-9)
    assert band.m == 46 * 132 and band.dim == 3 * band.m
    x = np.arange(model.dim, dtype=float)
    u, v, h = model.unpack(x)
    z = band.extract(x)
    assert np.array_equal(band.field_map(z, 2), h[10:56])           # same layout as swesphere
    assert np.array_equal(band.insert(np.zeros(model.dim), z)[band.state_index], z)


def test_wake_on_the_models_own_solid_body_flow():
    m, x0 = make_model("williamson_tc2", alpha_deg=0.0)
    band = Band.from_model(m)
    u, v, _ = m.unpack(x0)
    lag = 24 * 3600.0
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], band.radius), lag, lag, width=1)
    U0 = u[m.Nlat // 2].mean()                          # equatorial speed of the preset
    expected = -np.degrees(U0 * lag / band.radius) / (360.0 / m.Nlon)
    assert np.allclose(w.disp[:, 1], expected, atol=2e-3)
    assert np.allclose(w.disp[:, 0], 0.0, atol=1e-6)


def test_twin_construction(model):
    rng = np.random.default_rng(0)
    x0 = make_model("two_jets")[1]
    states = np.stack([x0 + rng.standard_normal(model.dim) * 0.0 for _ in range(3)])
    states[1] = model.propagate(x0, [0.0, 0.25 * DAY])
    states[2] = model.propagate(x0, [0.0, 0.5 * DAY])
    with Propagator(workers=1) as prop:
        twin = make_twin(prop, states, seed=1, n_members=3, t1_days=0.05, t2_days=0.05,
                         horizon_h=6)
    assert twin.ens.shape == (2, 3, model.dim) and list(twin.hours) == [0.0, 6.0]
    assert twin.t0 == pytest.approx(0.1 * DAY)
    assert not np.allclose(twin.truth[0], twin.free[0])       # background is not the truth
    band = Band.from_model(model)
    cells = draw_network(band, 0.1, rng)
    assert cells.size == round(0.1 * band.m) and np.all(np.diff(cells) > 0)
    y = observe_h(band, twin.truth[1], cells, 0.0, rng)
    assert np.array_equal(y, model.unpack(twin.truth[1])[2][10:56].ravel()[cells])
