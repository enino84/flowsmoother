"""Thin wrappers around swesphere 0.6.0 (no sponge, RK4, two_jets by default)."""
from __future__ import annotations

import os
from multiprocessing import get_context

import numpy as np

HOUR = 3600.0
DAY = 86400.0

_MODEL = None  # per-process model, created lazily (workers build their own)


def make_model(preset: str = "two_jets", **overrides):
    """Return ``(model, x0)`` for a swesphere preset. ``n_pole_rows`` stays 0."""
    from swesphere import presets

    return presets.make(preset, **overrides)


def _worker_init(preset, overrides):
    global _MODEL
    _MODEL = make_model(preset, **overrides)[0]


def _worker_run(args):
    x, t0, t1 = args
    return _MODEL.propagate(np.asarray(x, dtype=float), [t0, t1])


class Propagator:
    """Propagate many states in parallel, one model per worker process.

    The running time is always passed to ``propagate`` (trap 4 of the
    swesphere reference), so chained calls equal one uninterrupted run.
    """

    def __init__(self, preset="two_jets", workers=None, **overrides):
        self.preset, self.overrides = preset, overrides
        self.model = make_model(preset, **overrides)[0]
        self.workers = int(workers or os.cpu_count() or 1)
        self._pool = None
        if self.workers > 1:
            ctx = get_context("fork")
            self._pool = ctx.Pool(self.workers, _worker_init, (preset, overrides))

    def run(self, states, t0, t1):
        """states: (K, dim) -> (K, dim) at time t1."""
        states = np.atleast_2d(states)
        if t1 == t0:
            return states.copy()
        jobs = [(s, float(t0), float(t1)) for s in states]
        if self._pool is None:
            out = [self.model.propagate(np.asarray(s, float), [t0, t1]) for s in states]
        else:
            out = self._pool.map(_worker_run, jobs, chunksize=1)
        return np.asarray(out)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
