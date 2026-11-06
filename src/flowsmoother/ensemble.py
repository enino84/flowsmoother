"""Twin experiment: truth, background (free run), ensemble, observations.

Standard construction, every object dynamically consistent and at the same
instant ``t0``::

    t0-T1-T2            t0-T2                 t0            t0+lag
       |-------- T1 ------|-------- T2 --------|----- lag -----|
    truth  ---------------------------------------------------->  observations
    background = truth + noise -------------------------------->  free run
                        members = background + noise_k -------->  ensemble

The noise is added once. It is balanced: ``scale * (S_i - S_j)``, a scaled
difference of two record states (white noise would radiate as gravity waves).
After ``t0`` the ensemble is only ever propagated and analysed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Band
from .model import DAY, HOUR, Propagator


@dataclass
class Twin:
    t0: float                       # model time of t0 (s)
    hours: np.ndarray               # (T,) hours after t0 of the stored states
    truth: np.ndarray               # (T, dim)
    free: np.ndarray                # (T, dim) background propagated, never analysed
    ens: np.ndarray                 # (T, N, dim)
    draws: dict = field(default_factory=dict)

    def at(self, hour):
        k = int(np.where(self.hours == hour)[0][0])
        return self.truth[k], self.free[k], self.ens[k]

    def mean_wind(self, model, n_members):
        """Ensemble-mean (u, v) of the first ``n_members`` at every stored time."""
        mean = self.ens[:, :n_members].mean(axis=1)
        fs = model.field_size
        shape = (model.Nlat, model.Nlon)
        u = mean[:, :fs].reshape((-1,) + shape)
        v = mean[:, fs:2 * fs].reshape((-1,) + shape)
        return self.t0 + self.hours * HOUR, u, v


def make_twin(prop: Propagator, states, seed, n_members, scale=0.05,
              t1_days=2.0, t2_days=2.0, horizon_h=24, every_h=6) -> Twin:
    """Build truth, free run and ensemble at ``t0`` and carry them ``horizon_h``.

    states  (n, dim) record states at one time (independent enough draws).
    """
    rng = np.random.default_rng(seed)
    n = len(states)
    k_truth = int(rng.integers(n))
    p, q = rng.choice(n, size=2, replace=False)
    ij = np.array([rng.choice(n, size=2, replace=False) for _ in range(n_members)])
    S = lambda idx: np.asarray(states[int(idx)], dtype=float)  # noqa: E731
    truth = S(k_truth)
    background = truth + scale * (S(p) - S(q))
    t = 0.0
    truth, background = prop.run(np.stack([truth, background]), t, t + t1_days * DAY)
    t += t1_days * DAY
    members = np.stack([background + scale * (S(i) - S(j)) for i, j in ij])
    allst = prop.run(np.concatenate([[truth, background], members]), t, t + t2_days * DAY)
    t += t2_days * DAY
    hours = np.arange(0, horizon_h + every_h, every_h, dtype=float)
    out = [allst]
    for h0, h1 in zip(hours[:-1], hours[1:]):
        allst = prop.run(allst, t + h0 * HOUR, t + h1 * HOUR)
        out.append(allst)
    out = np.stack(out)                                   # (T, N + 2, dim)
    return Twin(t0=t, hours=hours, truth=out[:, 0], free=out[:, 1], ens=out[:, 2:],
                draws=dict(truth=k_truth, background=[int(p), int(q)], members=ij.tolist(),
                           scale=scale, t1_days=t1_days, t2_days=t2_days))


def hourly_truth(model, x0, t0, hours):
    """Truth every hour from ``t0`` (for animations), ``(hours + 1, dim)``."""
    out = [np.asarray(x0, dtype=float)]
    for k in range(int(hours)):
        out.append(model.propagate(out[-1], [t0 + k * HOUR, t0 + (k + 1) * HOUR]))
    return np.stack(out)


def draw_network(band: Band, fraction, rng):
    """Random observation sites: ``round(fraction * m)`` distinct cells, sorted."""
    n = max(1, int(round(fraction * band.m)))
    return np.sort(rng.choice(band.m, size=n, replace=False))


def observe_h(band: Band, x_truth, cells, sigma, rng):
    """Depth of the truth at the cells plus Gaussian error of sd ``sigma`` (m)."""
    h = band.extract(x_truth)[2 * band.m + cells]
    return h + sigma * rng.standard_normal(cells.size)
