"""Climatological record with cross-time pairs, and what is derived from it."""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .model import DAY, HOUR


def build_pair_record(model, x0, spinup_days, n_snapshots, every_days,
                      offsets_h=(0, 6, 12, 24), verbose=False):
    """Run once and keep ``(n_snapshots, len(offsets_h), dim)`` float32.

    Entry ``[k, j]`` is the state ``offsets_h[j]`` hours after snapshot ``k``.
    The offsets must fit inside the snapshot spacing so one trajectory serves.
    """
    offsets = [float(h) * HOUR for h in offsets_h]
    if offsets[0] != 0.0 or sorted(offsets) != offsets:
        raise ValueError("offsets_h must start at 0 and increase")
    if offsets[-1] >= every_days * DAY:
        raise ValueError("offsets must be shorter than the snapshot spacing")
    rec = np.empty((n_snapshots, len(offsets), model.dim), dtype=np.float32)
    t = 0.0
    x = model.propagate(np.asarray(x0, float), [t, t + spinup_days * DAY])
    t += spinup_days * DAY
    for k in range(n_snapshots):
        for j, off in enumerate(offsets):
            target = spinup_days * DAY + k * every_days * DAY + off
            x = model.propagate(x, [t, target])
            t = target
            rec[k, j] = x
        if verbose and (k % 20 == 0 or k == n_snapshots - 1):
            print(f"  snapshot {k + 1}/{n_snapshots}", flush=True)
    return rec


class Record:
    """Memory-mapped record plus the per-field anomaly scales."""

    def __init__(self, path):
        path = pathlib.Path(path)
        self.data = np.load(path / "record.npy", mmap_mode="r")
        self.meta = json.loads((path / "record.json").read_text())
        self.offsets_h = [float(h) for h in self.meta["offsets_h"]]
        self.n, _, self.dim = self.data.shape
        self.fs = self.dim // 3

    def states(self, offset_h=0.0):
        return self.data[:, self.offsets_h.index(float(offset_h))]

    def field_sd(self, band=None):
        """Climatological anomaly sd of u, v, h (about the record mean), (3,)."""
        s = np.asarray(self.states(0.0), dtype=float)
        an = s - s.mean(axis=0)
        out = np.empty(3)
        for f in range(3):
            blk = an[:, f * self.fs:(f + 1) * self.fs]
            if band is not None:
                blk = blk[:, band.field_index]
            out[f] = np.sqrt(np.mean(blk ** 2))
        return out

    def joint_anomalies(self, band, lag_h, sd):
        """Normalized band anomalies of ``[S(t), S(t+lag)]``, (n, 6m) float32."""
        out = np.empty((self.n, 2 * band.dim), dtype=np.float32)
        for tau, off in enumerate((0.0, float(lag_h))):
            s = np.asarray(self.states(off), dtype=float)
            z = band.extract(s - s.mean(axis=0)) / band.scale_vector(sd)
            out[:, tau * band.dim:(tau + 1) * band.dim] = z
        return out


# --------------------------------------------------------------- climate prior
def clim_centres(ZcT, group, alpha_clim=0.1):
    """Row coefficients fitted on the climatological anomalies: ``beta^clim``.

    ZcT is the padded record anomaly matrix (``rows.pad_anomalies``) of the
    same state the group indexes (single time or joint). The fit is a ridge
    regression with a small relative ridge ``alpha_clim``; with a few hundred
    record states and wide rows the regression is close to saturated, so the
    ridge is what keeps the centre from fitting the record's sampling noise.
    """
    from .rows import fit_rows

    beta, _ = fit_rows(ZcT, group.targets, group.pred, group.plen, alpha_clim)
    return beta


def shift_same_time_centre(beta_a, group_a, group_b):
    """Centre for an ``x_b`` row that carries climate only on its same-time part.

    The same-time predecessors of an ``x_b`` row are those of the matching
    ``x_a`` row shifted by ``3m``; being the larger indices they sit after the
    cross-time ones in the sorted list. The cross-time part gets centre zero.
    """
    out = np.zeros(group_b.pred.shape)
    n_cross = group_b.plen - group_b.n_same
    P = beta_a.shape[1]
    cols = n_cross[:, None] + np.arange(P)[None, :]
    ok = np.arange(P)[None, :] < group_a.plen[:, None]
    rows = np.broadcast_to(np.arange(out.shape[0])[:, None], cols.shape)
    out[rows[ok], cols[ok]] = beta_a[ok]
    return out
