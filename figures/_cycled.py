"""Helpers for the figures of the cycled experiment (E-S2)."""
import numpy as np

from flowsmoother import plotting as P
from flowsmoother.summary import levels, select


def arm_style(arm, arms):
    """Colour by family, line style to tell members of a family apart."""
    fam = arm_style_family(arm)
    same = [x for x in arms if arm_style_family(x) == fam]
    return P.METHOD[fam], ["-", "--", ":", "-."][same.index(arm) % 4]


def arm_style_family(arm):
    if arm.startswith("letks"):
        return "LETKS"
    if arm.startswith("filter"):
        return "EnKF-MC (filter)"
    s = arm.split("-")[1]
    if s.startswith("patch"):
        return "EnKS-MC patch"
    return "EnKS-MC square" if s == "square" else "EnKS-MC wake"


def series(rows, metric, cycles, **where):
    """(seeds, cycles) array of a metric, NaN where a run stopped."""
    sub = select(rows, **where)
    seeds = levels(sub, "seed")
    out = np.full((len(seeds), cycles), np.nan)
    for r in sub:
        out[seeds.index(r["seed"]), r["cycle"] - 1] = r[metric]
    return out


def best_inflation(rows, arm, metric, cycles, skip, **where):
    """Inflation with the lowest mean of the metric over the scored cycles.

    A run that stopped (diverged) counts as worse than any that finished.
    """
    best, score = None, np.inf
    for infl in levels(select(rows, arm=arm, **where), "inflation"):
        s = series(rows, metric, cycles, arm=arm, inflation=float(infl), **where)[:, skip:]
        val = np.inf if np.isnan(s).any() else float(np.mean(s))
        if best is None or val < score:
            best, score = infl, val
    return best, score
