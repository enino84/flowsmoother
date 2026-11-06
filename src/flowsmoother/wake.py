"""Backward trajectories of the ensemble-mean wind and the cells of the wake.

The wake of a cell is the set of cells within ``width`` (Chebyshev) of the
path followed, backward in time over the lag, by the parcel that arrives at
the cell. Those are the cross-time predecessors of the ``h`` rows.

Sign conventions, checked by the tests: with a westerly wind (``u > 0``) the
parcel came from the west, i.e. from *smaller* column indices; with a
southerly wind (``v > 0``) it came from the south, i.e. from *larger* row
indices, because rows are stored north to south.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Band, pad_rows, unique_rows


def _bilinear(fld, rho, kap):
    """Bilinear interpolation of ``fld (nlat, nlon)`` at fractional (row, col)."""
    nlat, nlon = fld.shape
    rho = np.clip(rho, 0.0, nlat - 1.0)
    i0 = np.minimum(np.floor(rho).astype(int), nlat - 2)
    fi = rho - i0
    kap = np.mod(kap, nlon)
    j0 = np.floor(kap).astype(int) % nlon
    fj = kap - np.floor(kap)
    j1 = (j0 + 1) % nlon
    top = fld[i0, j0] * (1 - fj) + fld[i0, j1] * fj
    bot = fld[i0 + 1, j0] * (1 - fj) + fld[i0 + 1, j1] * fj
    return top * (1 - fi) + bot * fi


class MeanWind:
    """Ensemble-mean wind, piecewise linear in time between stored states."""

    def __init__(self, times, u_seq, v_seq, radius=6.371e6):
        self.times = np.asarray(times, dtype=float)
        self.u = np.asarray(u_seq, dtype=float)       # (T, nlat, nlon)
        self.v = np.asarray(v_seq, dtype=float)
        if self.u.shape[0] != self.times.size or np.any(np.diff(self.times) <= 0):
            raise ValueError("need one wind field per strictly increasing time")
        self.nlat, self.nlon = self.u.shape[1:]
        self.radius = radius
        self.dlat = 180.0 / self.nlat
        self.dlon = 360.0 / self.nlon

    def index_velocity(self, rho, kap, t):
        """Velocity in grid-index units per second: ``(d rho/dt, d kap/dt)``."""
        t = float(np.clip(t, self.times[0], self.times[-1]))
        k = int(np.clip(np.searchsorted(self.times, t, side="right") - 1,
                        0, self.times.size - 2)) if self.times.size > 1 else 0
        if self.times.size > 1:
            w = (t - self.times[k]) / (self.times[k + 1] - self.times[k])
            u = (1 - w) * _bilinear(self.u[k], rho, kap) + w * _bilinear(self.u[k + 1], rho, kap)
            v = (1 - w) * _bilinear(self.v[k], rho, kap) + w * _bilinear(self.v[k + 1], rho, kap)
        else:
            u, v = _bilinear(self.u[0], rho, kap), _bilinear(self.v[0], rho, kap)
        lat = np.radians(90.0 - np.clip(rho, 0.0, self.nlat - 1.0) * self.dlat)
        cosl = np.maximum(np.cos(lat), np.cos(np.radians(85.0)))
        dkap = np.degrees(u / (self.radius * cosl)) / self.dlon      # east = +col
        drho = -np.degrees(v / self.radius) / self.dlat              # north = -row
        return drho, dkap


def backtrack(wind: MeanWind, rho0, kap0, t_end, lag, substep=1200.0, speed=1.0):
    """RK4 backward trajectories from ``(rho0, kap0)`` at ``t_end`` over ``lag`` s.

    Returns ``rho, kap`` of shape ``(points, S + 1)``; column 0 is the start
    (time ``t_end``), the last column is the position at ``t_end - lag``.
    ``speed`` multiplies the wind (1 = parcels; below 1 = a pattern that moves
    slower than the air it sits in, see ``verify_predecessors.py``).
    """
    n = max(1, int(np.ceil(lag / substep)))
    h = lag / n
    rho = np.empty((np.size(rho0), n + 1))
    kap = np.empty_like(rho)
    rho[:, 0], kap[:, 0] = rho0, kap0
    r, k = np.array(rho0, dtype=float), np.array(kap0, dtype=float)
    def vel(rr, kk, tt):
        a, b = wind.index_velocity(rr, kk, tt)
        return speed * a, speed * b

    for s in range(n):
        t = t_end - s * h
        a1, b1 = vel(r, k, t)
        a2, b2 = vel(r - 0.5 * h * a1, k - 0.5 * h * b1, t - 0.5 * h)
        a3, b3 = vel(r - 0.5 * h * a2, k - 0.5 * h * b2, t - 0.5 * h)
        a4, b4 = vel(r - h * a3, k - h * b3, t - h)
        r = r - h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
        k = k - h * (b1 + 2 * b2 + 2 * b3 + b4) / 6.0
        rho[:, s + 1], kap[:, s + 1] = r, k
    return rho, kap


@dataclass
class Wake:
    cells: np.ndarray        # (m, W) cells of the wake, -1 padded, sorted
    count: np.ndarray        # (m,) number of cells
    length: np.ndarray       # (m,) path length in cells
    disp: np.ndarray         # (m, 2) end minus start in (rows, cols)
    rho: np.ndarray          # (m, S+1) grid row along the path (full-grid index)
    kap: np.ndarray          # (m, S+1) grid column along the path (unwrapped)
    hit_cap: np.ndarray      # (m,) path cut by the length cap
    left_band: np.ndarray    # (m,) path cut at the band edge


def wake_cells(band: Band, wind: MeanWind, t_end, lag, width=2, max_cells=40.0,
               substep=1200.0, speed=1.0) -> Wake:
    """Wake of every band cell for a forecast from ``t_end - lag`` to ``t_end``."""
    rho0 = (band.r0 + band.cell_i).astype(float)
    kap0 = band.cell_j.astype(float)
    rho, kap = backtrack(wind, rho0, kap0, t_end, lag, substep, speed)
    step = np.hypot(np.diff(rho, axis=1), np.diff(kap, axis=1))
    path = np.concatenate([np.zeros((band.m, 1)), np.cumsum(step, axis=1)], axis=1)
    bi = np.rint(rho).astype(int) - band.r0            # band row of every point
    bj = np.rint(kap).astype(int)
    inside = (bi >= 0) & (bi < band.nr)
    under = path <= max_cells
    ok = np.logical_and.accumulate(inside & under, axis=1)   # cut once and for all
    last = ok.sum(axis=1) - 1                                 # ok[:, 0] is always True
    rows = np.arange(band.m)
    offs = [(di, dj) for di in range(-width, width + 1) for dj in range(-width, width + 1)]
    cand = np.empty((band.m, ok.shape[1] * len(offs)), dtype=np.int64)
    val = np.empty(cand.shape, dtype=bool)
    for q, (di, dj) in enumerate(offs):
        i = bi + di
        j = (bj + dj) % band.nc
        sl = slice(q * ok.shape[1], (q + 1) * ok.shape[1])
        cand[:, sl] = i * band.nc + j
        val[:, sl] = ok & (i >= 0) & (i < band.nr)
    key, keep = unique_rows(cand, val)
    cells, count = pad_rows(key, keep, -1)
    disp = np.stack([rho[rows, last] - rho[:, 0], kap[rows, last] - kap[:, 0]], axis=1)
    cut = ~ok.all(axis=1)
    first_bad = np.minimum(last + 1, ok.shape[1] - 1)
    return Wake(cells=cells, count=count, length=path[rows, last], disp=disp,
                rho=rho, kap=kap,
                hit_cap=cut & ~under[rows, first_bad],
                left_band=cut & ~inside[rows, first_bad])


def wake_speed(struct):
    """Speed factor encoded in a structure name: ``wake`` -> 1, ``wake50`` -> 0.5."""
    digits = struct[5:] if struct.startswith("patch") else struct[4:]
    return float(digits) / 100.0 if digits else 1.0


def wakes_for(structs, band, wind, t_end, lag, width=2, max_cells=40.0):
    """One Wake per distinct speed among the requested structures, keyed ``wake<P>``."""
    out = {}
    for s in structs:
        if s.startswith(("wake", "patch")):
            key = "wake" + (s[5:] if s.startswith("patch") else s[4:])
            if key not in out:
                out[key] = wake_cells(band, wind, t_end, lag, width, max_cells,
                                      speed=wake_speed(s))
    return out
