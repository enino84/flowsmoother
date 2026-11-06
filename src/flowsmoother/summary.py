"""Statistics over seeds for the metrics tables (used by summaries and figures)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

FAMILIES = {
    "LETKS": lambda r: r["kind"] == "letks",
    "EnKF-MC (filter)": lambda r: r["kind"] == "filter" and r["prior"] == "off",
    "EnKS-MC square": lambda r: r["kind"] == "enks" and r["struct"] == "square" and r["prior"] == "off",
    "EnKS-MC patch": lambda r: r["kind"] == "enks" and r["struct"] == "patch" and r["prior"] == "off",
    "EnKS-MC wake": lambda r: r["kind"] == "enks" and r["struct"] == "wake" and r["prior"] == "off",
    "EnKS-MC square + clim": lambda r: r["kind"] == "enks" and r["struct"] == "square" and r["prior"] != "off",
    "EnKS-MC wake + clim": lambda r: r["kind"] == "enks" and r["struct"] == "wake" and r["prior"] != "off",
}


def select(rows, **where):
    """Rows matching every ``key=value`` (floats compared with tolerance)."""
    def ok(r):
        for k, v in where.items():
            x = r.get(k)
            if isinstance(v, float) or isinstance(x, float):
                if x is None or abs(float(x) - float(v)) > 1e-9:
                    return False
            elif x != v:
                return False
        return True
    return [r for r in rows if ok(r)]


def by_arm(rows, metric):
    """arm -> {seed: value}."""
    out = defaultdict(dict)
    for r in rows:
        out[r["arm"]][r["seed"]] = r[metric]
    return out


def mean_se(values):
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return np.nan, np.nan, 0
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    return float(v.mean()), float(se), int(v.size)


def best_arm(rows, family, tune=("ratio_a_tot", "ratio_b_tot")):
    """Arm of a family with the lowest seed-mean of the tuning metrics.

    The same data choose the configuration and report it, so the value is
    optimistic; every family gets the same treatment.
    """
    cand = [r for r in rows if FAMILIES[family](r)]
    score = defaultdict(list)
    for r in cand:
        score[r["arm"]].append(np.mean([r[m] for m in tune]))
    if not score:
        return None
    return min(score, key=lambda a: np.mean(score[a]))


def paired(rows, arm_a, arm_b, metric):
    """Per-seed differences ``arm_a - arm_b`` on the seeds both have."""
    a, b = by_arm(rows, metric)[arm_a], by_arm(rows, metric)[arm_b]
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] - b[s] for s in seeds]), seeds


def swap_struct(arm, struct):
    """Same configuration with another cross-time structure."""
    parts = arm.split("-")
    parts[1] = struct
    return "-".join(parts)


def levels(rows, key):
    return sorted({r[key] for r in rows if r.get(key) is not None})
