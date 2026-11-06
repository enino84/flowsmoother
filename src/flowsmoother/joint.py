"""Predecessor graph of the joint state ``z = [x_a, x_b]`` (lag-1 smoother).

===========  =============================  =====================================
row          same time                      earlier time (cross-time block)
===========  =============================  =====================================
x_a (u,v,h)  square r, three fields         --
h   in x_b   square r, three fields         wake (or square, or matched patch),
                                            three fields, own cell included
u,v in x_b   square r, three fields         square r, three fields
===========  =============================  =====================================

Within one time only lower indices count (field-major, row-major order), so a
``u`` row sees the half square of ``u``, a ``v`` row the full square of ``u``
and half of ``v``, an ``h`` row the full squares of ``u, v`` and half of ``h``.
Every ``x_a`` index is lower than every ``x_b`` index, so cross-time
neighbourhoods are used whole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Band, pad_rows
from .rows import assemble, fit_rows

GROUPS_JOINT = ("a_u", "a_v", "a_h", "b_u", "b_v", "b_h")


@dataclass
class RowGroup:
    name: str
    targets: np.ndarray       # (R,) indices of the rows
    pred: np.ndarray          # (R, P) predecessors, padded with ``dim``
    plen: np.ndarray          # (R,)
    n_same: np.ndarray        # (R,) how many predecessors are same-time


@dataclass
class Graph:
    dim: int
    groups: dict = field(default_factory=dict)

    def predecessors(self, index):
        """Sorted predecessors of one row (for inspection and tests)."""
        for g in self.groups.values():
            k = np.searchsorted(g.targets, index)
            if k < g.targets.size and g.targets[k] == index:
                return g.pred[k, :g.plen[k]]
        raise KeyError(index)

    def mean_predecessors(self):
        return sum(g.plen.sum() for g in self.groups.values()) / self.dim


def _same_time_candidates(band: Band, r: int, f: int):
    """Candidates (band-state indices) of field ``f`` rows and their validity."""
    sq = band.square(r)
    cand = np.concatenate([g * band.m + sq for g in range(3)], axis=1)
    target = f * band.m + np.arange(band.m)
    valid = np.tile(sq >= 0, (1, 3)) & (cand < target[:, None])
    return cand, valid


def _cross_candidates(band: Band, cells):
    """Three fields of the given cells at the earlier time (indices 0..3m-1)."""
    cand = np.concatenate([g * band.m + cells for g in range(3)], axis=1)
    return cand, np.tile(cells >= 0, (1, 3))


def build_filter_graph(band: Band, r: int) -> Graph:
    """Single-time graph (the EnKF-MC filter): squares of radius ``r``."""
    graph = Graph(dim=band.dim)
    for f, name in enumerate(("f_u", "f_v", "f_h")):
        cand, valid = _same_time_candidates(band, r, f)
        pred, plen = pad_rows(cand, valid, graph.dim)
        graph.groups[name] = RowGroup(name, f * band.m + np.arange(band.m), pred, plen, plen)
    return graph


def build_joint_graph(band: Band, r: int, cross_h, cross_uv=None) -> Graph:
    """Joint graph of ``[x_a, x_b]``.

    cross_h   (m, W) cells at the earlier time for the ``h`` rows of ``x_b``
              (-1 padded): a wake, a square or a matched patch; ``None`` for no
              cross-time link.
    cross_uv  same for the ``u, v`` rows of ``x_b``; default the square ``r``.
    """
    m, off = band.m, band.dim
    graph = Graph(dim=2 * band.dim)
    if cross_uv is None:
        cross_uv = band.square(r)
    for f, fname in enumerate(("u", "v", "h")):
        cand, valid = _same_time_candidates(band, r, f)
        pred, plen = pad_rows(cand, valid, graph.dim)
        graph.groups["a_" + fname] = RowGroup("a_" + fname, f * m + np.arange(m), pred, plen, plen)
        cells = cross_h if fname == "h" else cross_uv
        parts_c, parts_v = [cand + off], [valid]
        if cells is not None:
            cc, cv = _cross_candidates(band, cells)
            parts_c.append(cc)
            parts_v.append(cv)
        pred_b, plen_b = pad_rows(np.concatenate(parts_c, axis=1),
                                  np.concatenate(parts_v, axis=1), graph.dim)
        graph.groups["b_" + fname] = RowGroup("b_" + fname, off + f * m + np.arange(m),
                                              pred_b, plen_b, plen)
    return graph


def check_graph(graph: Graph, band: Band | None = None):
    """Structural invariants every graph must satisfy; raises on violation."""
    seen = np.zeros(graph.dim, dtype=int)
    for g in graph.groups.values():
        seen[g.targets] += 1
        valid = g.pred < graph.dim
        if not np.array_equal(valid.sum(axis=1), g.plen):
            raise AssertionError(f"{g.name}: plen does not match the padding")
        if np.any(valid[:, 1:] & ~valid[:, :-1]):
            raise AssertionError(f"{g.name}: padding is not at the end")
        if np.any((g.pred >= g.targets[:, None]) & valid):
            raise AssertionError(f"{g.name}: predecessor not lower than its row")
        d = np.diff(g.pred, axis=1)
        if np.any((d <= 0) & valid[:, 1:]):
            raise AssertionError(f"{g.name}: predecessors not sorted or repeated")
        if band is not None and graph.dim == 2 * band.dim:
            if g.name.startswith("a_") and np.any(valid & (g.pred >= band.dim)):
                raise AssertionError(f"{g.name}: x_a row with a predecessor in x_b")
            if g.name.startswith("b_"):
                same = (valid & (g.pred >= band.dim)).sum(axis=1)
                if not np.array_equal(same, g.n_same):
                    raise AssertionError(f"{g.name}: same-time count mismatch")
    if not np.all(seen == 1):
        raise AssertionError("every row must belong to exactly one group")
    return True


def fit_graph(XT, graph: Graph, alpha, centres=None, cache=None, cache_key=None,
              sigma="insample"):
    """Fit every group and assemble the factor.

    centres    dict group name -> (R, P) ridge centre (missing = zero).
    cache      dict reused across graphs that share groups; ``cache_key`` maps a
               group name to a hashable key (same key = same rows, same
               predecessors, same ridge, same centre).
    """
    parts = []
    for name, g in graph.groups.items():
        key = None if cache is None or cache_key is None else cache_key.get(name)
        if key is not None and key in cache:
            beta, s2 = cache[key]
        else:
            c = None if centres is None else centres.get(name)
            beta, s2 = fit_rows(XT, g.targets, g.pred, g.plen, alpha, centre=c, sigma=sigma)
            if key is not None:
                cache[key] = (beta, s2)
        parts.append((g.targets, g.pred, beta, s2))
    return assemble(graph.dim, parts)
