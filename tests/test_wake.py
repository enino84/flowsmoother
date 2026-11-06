"""Backward wakes: direction, length, wrap-around, cuts."""
import numpy as np
import pytest

from flowsmoother.wake import MeanWind, backtrack, wake_cells

A = 6.371e6


def winds(band, u, v):
    shape = (band.nlat, band.nlon)
    return np.broadcast_to(u, shape).copy(), np.broadcast_to(v, shape).copy()


def solid_body(band, U):
    lat = np.radians(band.lat_deg)[:, None]
    return winds(band, U * np.cos(lat), 0.0)


def test_solid_body_displacement_is_analytic(band):
    U, lag = 40.0, 12 * 3600.0
    u, v = solid_body(band, U)
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], A), lag, lag, width=1)
    expected = -np.degrees(U * lag / A) / (360.0 / band.nlon)        # columns, westward
    assert np.allclose(w.disp[:, 1], expected, atol=1e-9)
    assert np.allclose(w.disp[:, 0], 0.0, atol=1e-12)
    assert np.allclose(w.length, abs(expected), atol=1e-9)


def test_westerly_wind_gives_a_wake_to_the_west(band):
    lag = 48 * 3600.0
    u, v = solid_body(band, 60.0)
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], A), lag, lag, width=0)
    shift = int(round(-w.disp[0, 1]))
    assert shift >= 3
    c = 6 * band.nc + 8
    cols = band.cell_j[w.cells[c][w.cells[c] >= 0]]
    rows = band.cell_i[w.cells[c][w.cells[c] >= 0]]
    assert set(rows) == {6}
    assert set(cols) == {(8 - k) % band.nc for k in range(shift + 1)}   # upstream = west


def test_southerly_wind_gives_a_wake_to_the_south(band):
    lag = 12 * 3600.0
    u, v = winds(band, 0.0, 15.0)                       # blowing northward
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], A), lag, lag, width=0)
    c = 4 * band.nc + 3
    expected = np.degrees(15.0 * lag / A) / (180.0 / band.nlat)       # rows, southward (+)
    assert w.disp[c, 0] == pytest.approx(expected, rel=1e-9)
    rows = band.cell_i[w.cells[c][w.cells[c] >= 0]]
    assert rows.min() == 4 and rows.max() == 4 + int(round(expected))


def test_zero_wind_wake_is_the_square(band):
    u, v = winds(band, 0.0, 0.0)
    for width in (1, 2):
        w = wake_cells(band, MeanWind([0.0, 3600.0], [u, u], [v, v], A), 3600.0, 3600.0,
                       width=width)
        sq = band.square(width)
        for c in range(band.m):
            assert set(w.cells[c][w.cells[c] >= 0]) == set(sq[c][sq[c] >= 0])
        assert np.all(w.length == 0)


def test_wake_wraps_around_the_seam(band):
    lag = 24 * 3600.0
    u, v = solid_body(band, 60.0)
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], A), lag, lag, width=0)
    c = 6 * band.nc + 1
    cols = set(band.cell_j[w.cells[c][w.cells[c] >= 0]])
    assert {1, 0, band.nc - 1} <= cols


def test_wake_contains_own_cell_and_is_sorted_unique(band):
    rng = np.random.default_rng(0)
    u = 30.0 * rng.standard_normal((band.nlat, band.nlon))
    v = 10.0 * rng.standard_normal((band.nlat, band.nlon))
    w = wake_cells(band, MeanWind([0.0, 6 * 3600.0], [u, u], [v, v], A), 6 * 3600.0,
                   6 * 3600.0, width=2)
    for c in range(band.m):
        cells = w.cells[c][w.cells[c] >= 0]
        assert c in cells
        assert np.all(np.diff(cells) > 0)
        assert cells.size == w.count[c]
        assert cells.min() >= 0 and cells.max() < band.m
        sq = band.square(2)[c]
        assert set(sq[sq >= 0]) <= set(cells)            # at least the square around itself


def test_length_cap_and_band_edge(band):
    lag = 48 * 3600.0
    u, v = solid_body(band, 80.0)
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], A), lag, lag, width=1, max_cells=3.0)
    assert w.hit_cap.all() and np.all(w.length <= 3.0 + 1e-9) and not w.left_band.any()
    u, v = winds(band, 0.0, -40.0)                      # blowing southward: came from the north
    w = wake_cells(band, MeanWind([0.0, lag], [u, u], [v, v], A), lag, lag, width=1)
    top = np.where(band.cell_i == 0)[0]
    assert w.left_band[top].all()
    assert np.all(w.cells[w.cells >= 0] < band.m)
    assert np.all(band.cell_i[w.cells[top][w.cells[top] >= 0]] <= 1)


def test_time_interpolation_of_the_wind(band):
    """Wind growing linearly in time: displacement is that of the time mean."""
    lag = 6 * 3600.0
    u0, v0 = solid_body(band, 20.0)
    u1, _ = solid_body(band, 60.0)
    w = wake_cells(band, MeanWind([0.0, lag], [u0, u1], [v0, v0], A), lag, lag, width=0)
    expected = -np.degrees(40.0 * lag / A) / (360.0 / band.nlon)
    assert np.allclose(w.disp[:, 1], expected, atol=1e-9)


def test_backtrack_reverses_a_forward_step(band):
    """Going back with the wind reversed in sign returns to the start."""
    rng = np.random.default_rng(3)
    lat = np.radians(band.lat_deg)[:, None]
    lon = np.radians(np.arange(band.nlon) * 360.0 / band.nlon)[None, :]
    u = 30.0 * np.cos(lat) * (1 + 0.3 * np.sin(2 * lon))
    v = 8.0 * np.cos(lat) * np.cos(3 * lon)
    lag = 6 * 3600.0
    r0 = rng.uniform(band.r0 + 2, band.r1 - 2, 50)
    k0 = rng.uniform(0, band.nlon, 50)
    rb, kb = backtrack(MeanWind([0.0, lag], [u, u], [v, v], A), r0, k0, lag, lag, 600.0)
    rf, kf = backtrack(MeanWind([0.0, lag], [-u, -u], [-v, -v], A), rb[:, -1], kb[:, -1],
                       lag, lag, 600.0)
    assert np.allclose(rf[:, -1], r0, atol=1e-6)
    assert np.allclose(kf[:, -1], k0, atol=1e-6)


def test_speed_factor_scales_the_displacement(band):
    from flowsmoother.wake import wake_speed, wakes_for
    lag = 12 * 3600.0
    u, v = solid_body(band, 40.0)
    wind = MeanWind([0.0, lag], [u, u], [v, v], A)
    w = wakes_for(["square", "wake", "patch", "wake50", "patch50"], band, wind, lag, lag, width=1)
    assert set(w) == {"wake", "wake50"}
    assert np.allclose(w["wake50"].disp, 0.5 * w["wake"].disp, atol=1e-9)
    assert wake_speed("wake") == 1.0 and wake_speed("patch50") == 0.5
