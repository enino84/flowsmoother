"""Latitude band of the swesphere grid and its index bookkeeping.

Conventions (identical to swesphere): rows run north to south, columns west
to east and are periodic; the state is ``[u, v, h]``, each field row-major.

Inside the band a *cell* is ``c = i * nc + j`` (``i`` band row, ``j`` column),
a *band state* index is ``f * m + c`` (``f`` = 0, 1, 2 for u, v, h) and a
*joint* index is ``tau * 3m + f * m + c`` (``tau`` = 0 for the earlier time
``a``, 1 for the later time ``b``). A predecessor is always a lower index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FIELDS = ("u", "v", "h")


@dataclass
class Band:
    nlat: int
    nlon: int
    r0: int                      # first grid row of the band (inclusive)
    r1: int                      # last grid row of the band (inclusive)
    lat_deg: np.ndarray          # (nlat,) latitude of every grid row
    radius: float = 6.371e6
    nr: int = field(init=False)
    nc: int = field(init=False)
    m: int = field(init=False)
    dim: int = field(init=False)

    def __post_init__(self):
        if not (0 <= self.r0 <= self.r1 < self.nlat):
            raise ValueError("band rows outside the grid")
        self.nr = self.r1 - self.r0 + 1
        self.nc = self.nlon
        self.m = self.nr * self.nc
        self.dim = 3 * self.m
        ii, jj = np.divmod(np.arange(self.m), self.nc)
        self.cell_i, self.cell_j = ii, jj
        self.field_index = (self.r0 + ii) * self.nlon + jj
        fs = self.nlat * self.nlon
        self.state_index = np.concatenate([f * fs + self.field_index for f in range(3)])
        w = np.cos(np.radians(self.lat_deg[self.r0 + ii]))
        self.weights = w / w.sum()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_model(cls, model, lat_north=63.0, lat_south=-60.5):
        """Rows with ``lat_south <= lat <= lat_north``: 10..55 at L = 32.

        The model runs without polar sponge; the polar rows are left out of
        the estimate and of the score because ``u, v`` are multivalued at the
        pole row and the polar rows are rough (swesphere reference, 9 and 10).
        """
        lat = 90.0 - np.arange(model.Nlat) * 180.0 / model.Nlat
        rows = np.where((lat <= lat_north) & (lat >= lat_south))[0]
        return cls(model.Nlat, model.Nlon, int(rows[0]), int(rows[-1]), lat,
                   float(model.grid.a))

    # ------------------------------------------------------------- state i/o
    def extract(self, x):
        """Full state(s) ``(..., 3*nlat*nlon)`` -> band state(s) ``(..., 3m)``."""
        return np.asarray(x)[..., self.state_index]

    def insert(self, x_full, z):
        """Copy of ``x_full`` with the band replaced by ``z``."""
        out = np.array(x_full, dtype=float, copy=True)
        out[..., self.state_index] = z
        return out

    def scale_vector(self, sd):
        """Per-field scales ``(3,)`` -> ``(3m,)``."""
        return np.repeat(np.asarray(sd, dtype=float), self.m)

    def field_map(self, z, f):
        """Field ``f`` of a band state as a ``(nr, nc)`` map."""
        return np.asarray(z)[..., f * self.m:(f + 1) * self.m].reshape(
            np.shape(z)[:-1] + (self.nr, self.nc))

    def cell_lat(self):
        return self.lat_deg[self.r0 + self.cell_i]

    def cell_lon(self):
        return self.cell_j * 360.0 / self.nlon

    # --------------------------------------------------------- neighbourhoods
    def shift(self, di, dj):
        """Cell reached from every cell by ``(di, dj)``; -1 if it leaves the band."""
        i = self.cell_i + di
        j = (self.cell_j + dj) % self.nc
        return np.where((i >= 0) & (i < self.nr), i * self.nc + j, -1)

    def square(self, r):
        """Chebyshev square of radius ``r``: ``(m, (2r+1)^2)`` cells, -1 outside."""
        if 2 * r + 1 > self.nc:
            raise ValueError("square wider than the periodic direction")
        offs = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]
        return np.stack([self.shift(di, dj) for di, dj in offs], axis=1)

    def patch(self, counts, rmax=None):
        """Isotropic patch with a prescribed number of cells per cell.

        Takes, for cell ``c``, its ``counts[c]`` nearest in-band cells
        (Euclidean distance in grid units, ties broken by a fixed order).
        This is the size control of the wake: same number of predecessors,
        placed without knowledge of the flow. Returns ``(m, K)`` padded with -1.
        """
        counts = np.asarray(counts, dtype=int)
        if rmax is None:
            rmax = 1
            while self._min_inband(rmax) < counts.max() and 2 * rmax + 3 <= self.nc:
                rmax += 1
        offs = [(di, dj) for di in range(-rmax, rmax + 1) for dj in range(-rmax, rmax + 1)]
        offs.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, max(abs(o[0]), abs(o[1])), o))
        cand = np.stack([self.shift(di, dj) for di, dj in offs], axis=1)
        ok = cand >= 0
        rank = np.cumsum(ok, axis=1)
        keep = ok & (rank <= counts[:, None])
        return pad_rows(cand, keep, -1)[0]

    def _min_inband(self, r):
        """Smallest number of in-band cells in a square of radius r (edge rows)."""
        return (2 * r + 1) * min(r + 1, self.nr)

    def chebyshev(self, c0, c1):
        """Chebyshev distance in cells (periodic in longitude)."""
        di = np.abs(self.cell_i[c0] - self.cell_i[c1])
        dj = np.abs(self.cell_j[c0] - self.cell_j[c1])
        dj = np.minimum(dj, self.nc - dj)
        return np.maximum(di, dj)


def pad_rows(cand, valid, sentinel):
    """Keep the valid entries of every row, sorted ascending, padded at the end.

    Returns ``(padded, count)`` with ``padded`` of shape ``(rows, count.max())``.
    """
    big = np.iinfo(np.int64).max
    key = np.where(valid, cand.astype(np.int64), big)
    key.sort(axis=1)
    count = valid.sum(axis=1)
    width = max(int(count.max()) if count.size else 0, 1)
    out = key[:, :width]
    out[out == big] = sentinel
    return out, count


def unique_rows(cand, valid):
    """Mark duplicates inside every row as invalid. Returns sorted cand and mask."""
    big = np.iinfo(np.int64).max
    key = np.where(valid, cand.astype(np.int64), big)
    key.sort(axis=1)
    ok = key != big
    ok[:, 1:] &= key[:, 1:] != key[:, :-1]
    return key, ok
