"""Joint precision of ``[x_a, x_b]`` under the different structures, with caches.

Structures of the cross-time block of the ``h`` rows of ``x_b``:

``square``    Chebyshev square of radius ``r`` (the control arm)
``wake``      cells of the backward wake of the ensemble-mean wind
``wake<P>``   same with the wind multiplied by ``P`` percent (``wake50``: half speed)
``patch``     isotropic patch with, cell by cell, as many cells as the wake
              (size control: same number of predecessors, placed without the flow);
              ``patch<P>`` is matched to ``wake<P>``

Everything else is identical across structures, so the only difference
between arms is where the cross-time predecessors of ``h`` are.

Priors on the row coefficients (the centre of the ridge):

``off``   zero (classical EnKF-MC)
``same``  climate on the same-time part of every row, zero on the cross-time part
``full``  climate on the whole row (cross-time coefficients from record pairs)
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from .clim import clim_centres, shift_same_time_centre
from .grid import Band
from .io import load_npz, save_npz
from .joint import build_filter_graph, build_joint_graph, check_graph, fit_graph
from .rows import pad_anomalies

PRIORS = ("off", "same", "full")


class JointEstimator:
    def __init__(self, band: Band, clim_anoms=None, alpha_clim=0.1, cache_dir=None,
                 cache_tag="", sigma="insample"):
        """clim_anoms: (M, 6m) normalized record anomalies of ``[S(t), S(t+lag)]``."""
        self.band = band
        self.alpha_clim = alpha_clim
        self.sigma = sigma
        self.ZcT = None if clim_anoms is None else pad_anomalies(clim_anoms)
        self.cache_dir = None if cache_dir is None else pathlib.Path(cache_dir)
        self.cache_tag = cache_tag
        self._graphs, self._fgraphs = {}, {}
        self._centres = {}            # flow-independent, kept across cases
        self._flow_centres = {}       # depend on the wake, reset with it
        self._fits = {}
        self.wake = None
        self.XT = self.XTb = None

    # ----------------------------------------------------------------- inputs
    def set_ensemble(self, Za, Zb):
        """Normalized band members at the two times, ``(N, 3m)`` each."""
        Z = np.concatenate([Za, Zb], axis=1)
        self.XT = pad_anomalies(Z - Z.mean(axis=0))
        self.XTb = pad_anomalies(Zb - Zb.mean(axis=0))
        self._fits = {}

    def set_wake(self, wake):
        """``wake``: a Wake (named ``wake``) or a dict name -> Wake."""
        self.wake = wake if isinstance(wake, dict) else {"wake": wake}
        self._graphs = {k: v for k, v in self._graphs.items() if k[0] == "square"}
        self._flow_centres = {}
        self._fits = {}

    # ----------------------------------------------------------------- graphs
    def cross_cells(self, struct, r):
        if struct == "square":
            return self.band.square(r)
        if self.wake is None:
            raise RuntimeError("set_wake first")
        if struct in self.wake:
            return self.wake[struct].cells
        if struct.startswith("patch") and "wake" + struct[5:] in self.wake:
            return self.band.patch(self.wake["wake" + struct[5:]].count)
        raise ValueError(f"unknown structure {struct!r}")

    def graph(self, struct, r):
        key = (struct, r)
        if key not in self._graphs:
            g = build_joint_graph(self.band, r, cross_h=self.cross_cells(struct, r))
            check_graph(g, self.band)
            self._graphs[key] = g
        return self._graphs[key]

    def filter_graph(self, r):
        if r not in self._fgraphs:
            g = build_filter_graph(self.band, r)
            check_graph(g, self.band)
            self._fgraphs[r] = g
        return self._fgraphs[r]

    # ---------------------------------------------------------------- centres
    def _disk(self, name):
        if self.cache_dir is None:
            return None
        tag = hashlib.md5(f"{self.cache_tag}|{self.alpha_clim}|{name}".encode()).hexdigest()[:12]
        return self.cache_dir / f"centre_{name}_{tag}.npz"

    def _clim_centre(self, graph, gname, key, flow):
        store = self._flow_centres if flow else self._centres
        if key in store:
            return store[key]
        if self.ZcT is None:
            raise RuntimeError("no climatology given: prior not available")
        path = None if flow else self._disk("_".join(str(k) for k in key))
        if path is not None and path.exists():
            beta = load_npz(path)["beta"].astype(float)
        else:
            beta = clim_centres(self.ZcT, graph.groups[gname], self.alpha_clim)
            if path is not None:
                save_npz(path, beta=beta)
        store[key] = beta
        return beta

    def centres(self, struct, r, prior):
        if prior == "off":
            return None
        g = self.graph(struct, r)
        out = {}
        for f in "uvh":
            a = self._clim_centre(g, "a_" + f, ("a", f, r), flow=False)
            out["a_" + f] = a
            if prior == "same":
                out["b_" + f] = shift_same_time_centre(a, g.groups["a_" + f], g.groups["b_" + f])
            elif f == "h":
                flow = struct != "square"
                out["b_h"] = self._clim_centre(g, "b_h", ("b_h", struct, r), flow=flow)
            else:
                out["b_" + f] = self._clim_centre(g, "b_" + f, ("b", f, r), flow=False)
        return out

    # ---------------------------------------------------------------- factors
    def factor(self, struct, r, alpha, prior="off"):
        """Factor of the joint precision; shared groups are fitted once."""
        g = self.graph(struct, r)
        pa = prior != "off"
        keys = {"a_" + f: ("a", f, r, alpha, pa) for f in "uvh"}
        keys.update({"b_" + f: ("b", f, r, alpha, prior) for f in "uv"})
        keys["b_h"] = ("b_h", struct, r, alpha, prior)
        return fit_graph(self.XT, g, alpha, centres=self.centres(struct, r, prior),
                         cache=self._fits, cache_key=keys, sigma=self.sigma)

    def filter_factor(self, r, alpha, prior="off"):
        """Single-time factor on ``x_b`` (the EnKF-MC filter, no cross-time block)."""
        g = self.filter_graph(r)
        centres = None
        if prior != "off":
            jg = self.graph("square", r)
            centres = {"f_" + f: self._clim_centre(jg, "a_" + f, ("a", f, r), flow=False)
                       for f in "uvh"}
        return fit_graph(self.XTb, g, alpha, centres=centres, sigma=self.sigma)
