"""Scores on the band: cos(latitude)-weighted, per field, in normalized units."""
from __future__ import annotations

import numpy as np

from .grid import Band

LABELS = ("u", "v", "h", "tot")


def rmse(band: Band, z, z_ref):
    """Weighted RMSE of u, v, h and the total (root of the mean of the three MSE).

    ``z`` and ``z_ref`` are band states in normalized units (each field divided
    by its climatological anomaly sd), so the three fields weigh the same.
    """
    d2 = (np.asarray(z) - np.asarray(z_ref)) ** 2
    mse = np.array([np.sum(band.weights * d2[f * band.m:(f + 1) * band.m]) for f in range(3)])
    return np.sqrt(np.append(mse, mse.mean()))


def spread(band: Band, Z):
    """Weighted ensemble spread of u, v, h and total, same convention as rmse."""
    var = np.var(np.asarray(Z), axis=0, ddof=1)
    v = np.array([np.sum(band.weights * var[f * band.m:(f + 1) * band.m]) for f in range(3)])
    return np.sqrt(np.append(v, v.mean()))


def cell_sq_error(band: Band, z, z_ref, f=2):
    """Squared error of one field per cell, ``(m,)``."""
    sl = slice(f * band.m, (f + 1) * band.m)
    return (np.asarray(z)[sl] - np.asarray(z_ref)[sl]) ** 2
