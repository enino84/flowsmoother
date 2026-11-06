"""Check the predecessor sets against the model itself.

Structure (always): builds the graphs at full size, runs the invariants of
``joint.check_graph`` and prints the predecessor counts per block.

Physics (needs a finished E-S1 run with its gallery seed): uses the large
reference ensemble to estimate the true cross-time correlation of depth,
``corr(h_b(i), h_a(j))``, in a window around every cell, and asks where it
lives:

  peak_in    fraction of cells whose largest |correlation| falls inside the set
  mass       fraction of the squared correlation (sampling floor removed) that
             the set captures
  peak_dist  Chebyshev distance from the peak to the end of the trajectory

for three sets: the wake, the square of radius r and the matched patch (same
size as the wake, no flow).

Those three describe the marginal correlation, which for a smooth field is
broad. What a Cholesky row needs is the conditional dependence, so the decisive
check is predictive:

  cv_<set>   out-of-sample unexplained fraction of the h row of x_b: the row is
             fitted on N members (the size used in the experiment) with each
             structure and scored on members it has not seen; mean over
             several disjoint training sets, best ridge of each structure

If the wake places the predecessors where the dependence is, cv_wake is below
cv_patch (same number of predecessors), and the more so the longer the wake.

And, since the smoother works through the covariance the precision implies:

  cov_<arm>  relative error of the implied cross-time covariance of depth,
             cov(h_b(i), h_a(window)), against the reference ensemble (members
             not used in the fit), for the factors saved by the gallery seed;
             cov_sample is the same for the raw N-member sample covariance

Usage
  python verify_predecessors.py --run pilot
"""
import argparse

import common
import numpy as np

from flowsmoother import io
from flowsmoother.grid import Band
from flowsmoother.joint import build_filter_graph, build_joint_graph, check_graph
from flowsmoother.model import make_model
from flowsmoother.rows import fit_rows, pad_anomalies

ALPHAS = (0.003, 0.03, 0.3)

BINS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 1e9)]


def structure_report(band, r, width):
    print(f"band rows {band.r0}..{band.r1}: {band.nr} x {band.nc} = {band.m} cells, "
          f"state {band.dim}, joint {2 * band.dim}")
    g = build_joint_graph(band, r, cross_h=band.square(width))
    check_graph(g, band)
    check_graph(build_filter_graph(band, r), band)
    print(f"invariants hold (r = {r}); predecessors per row, min / median / max:")
    for name, grp in g.groups.items():
        cross = grp.plen - grp.n_same
        print(f"  {name}: same-time {grp.n_same.min():3d} / {int(np.median(grp.n_same)):3d} / "
              f"{grp.n_same.max():3d}   cross-time {cross.min():3d} / "
              f"{int(np.median(cross)):3d} / {cross.max():3d}")


def physics(band, run, seed, lag, N, r, win_i, win_j):
    sdir = run / f"seed{seed:02d}"
    ref = io.load_npz(sdir / f"reference_lag{lag:02d}.npz")
    wk = io.load_npz(sdir / f"wake_lag{lag:02d}_N{N:03d}.npz")
    m = band.m
    ha = ref["Za"][:, 2 * m:].astype(float)
    hb = ref["Zb"][:, 2 * m:].astype(float)
    n = ha.shape[0]
    ha = (ha - ha.mean(0)) / ha.std(0, ddof=1)
    hb = (hb - hb.mean(0)) / hb.std(0, ddof=1)
    offs = [(di, dj) for di in range(-win_i, win_i + 1) for dj in range(-win_j, win_j + 1)]
    nb = np.stack([band.shift(di, dj) for di, dj in offs], axis=1)        # (m, K)
    C = np.zeros(nb.shape)
    for k in range(nb.shape[1]):
        ok = nb[:, k] >= 0
        C[ok, k] = np.einsum("nc,nc->c", hb[:, ok], ha[:, nb[ok, k]]) / (n - 1)
    floor = 1.0 / (n - 1)
    C2 = np.where(nb >= 0, np.maximum(C ** 2 - floor, 0.0), 0.0)
    peak = nb[np.arange(m), np.argmax(np.abs(C), axis=1)]
    sets = dict(wake=wk["cells"], square=band.square(r), patch=band.patch(wk["count"]))
    end_i = np.rint(wk["rho"][:, -1]).astype(int) - band.r0
    end_j = np.rint(wk["kap"][:, -1]).astype(int) % band.nc
    di = np.abs(band.cell_i[peak] - np.clip(end_i, 0, band.nr - 1))
    dj = np.abs(band.cell_j[peak] - end_j)
    peak_dist = np.maximum(di, np.minimum(dj, band.nc - dj))
    start_dist = band.chebyshev(peak, np.arange(m))
    rows = []
    for lo, hi in BINS + [(0.0, 1e9)]:
        sel = (wk["length"] >= lo) & (wk["length"] < hi)
        if not sel.any():
            continue
        row = dict(lag_h=lag, N=N, n_ref=n, length_bin=f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}",
                   cells=int(sel.sum()), wake_cells=float(wk["count"][sel].mean()),
                   peak_corr=float(np.abs(C).max(axis=1)[sel].mean()),
                   peak_to_traj_end=float(peak_dist[sel].mean()),
                   peak_to_own_cell=float(start_dist[sel].mean()))
        for name, cells in sets.items():
            inside = (nb[:, :, None] == cells[:, None, :]).any(axis=2) & (nb >= 0)
            row[f"peak_in_{name}"] = float(np.mean((peak[:, None] == cells).any(axis=1)[sel]))
            row[f"mass_{name}"] = float((C2 * inside)[sel].sum() / max(C2[sel].sum(), 1e-30))
        rows.append(row)
    return rows


def predictive(band, run, seed, lag, N, r, n_sets=5):
    """Out-of-sample unexplained fraction of the h rows of x_b, per structure."""
    sdir = run / f"seed{seed:02d}"
    ref = io.load_npz(sdir / f"reference_lag{lag:02d}.npz")
    Z = np.concatenate([ref["Za"], ref["Zb"]], axis=1).astype(float)
    n = Z.shape[0]
    n_sets = min(n_sets, (n // 2) // N)
    if n_sets < 1:
        return {}, None
    test = Z[n // 2:]
    test = test - test.mean(axis=0)
    wk = io.load_npz(sdir / f"wake_lag{lag:02d}_N{N:03d}.npz")
    cells = dict(square=band.square(r), wake=wk["cells"], patch=band.patch(wk["count"]))
    for extra in sorted(sdir.glob(f"wake?*_lag{lag:02d}_N{N:03d}.npz")):
        cells[extra.name.split("_")[0]] = io.load_npz(extra)["cells"]
    out = {}
    for name, cc in cells.items():
        grp = build_joint_graph(band, r, cross_h=cc).groups["b_h"]
        best = None
        for alpha in ALPHAS:
            frac = np.zeros(band.m)
            for k in range(n_sets):
                tr = Z[k * N:(k + 1) * N]
                beta, _ = fit_rows(pad_anomalies(tr - tr.mean(axis=0)), grp.targets, grp.pred,
                                   grp.plen, alpha)
                XT = pad_anomalies(test)
                for s0 in range(0, band.m, 256):
                    sl = slice(s0, s0 + 256)
                    pred = np.einsum("rp,rpn->rn", beta[sl], XT[grp.pred[sl]])
                    y = XT[grp.targets[sl]]
                    frac[sl] += ((y - pred) ** 2).sum(axis=1) / (y ** 2).sum(axis=1)
            frac /= n_sets
            if best is None or frac.mean() < best.mean():
                best = frac
        out[name] = best
    return out, wk["length"]


def implied_covariance(band, run, seed, lag, N, win_i, win_j, n_cells=400):
    """Relative error of the implied cov(h_b(i), h_a(window)) for saved factors."""
    from flowsmoother.analysis import TriangularSolver

    sdir = run / f"seed{seed:02d}"
    tag = f"lag{lag:02d}_N{N:03d}"
    files = sorted(sdir.glob(f"precision_{tag}_enks-*.npz"))
    if not files:
        return {}, None, None
    ref = io.load_npz(sdir / f"reference_lag{lag:02d}.npz")
    m, n = band.m, band.dim
    ha, hb = ref["Za"][:, 2 * m:].astype(float), ref["Zb"][:, 2 * m:].astype(float)
    if ha.shape[0] < 2 * N + 10:
        return {}, None, None
    rng = np.random.default_rng(0)
    inner = np.where((band.cell_i >= win_i) & (band.cell_i < band.nr - win_i))[0]
    cells = np.sort(rng.choice(inner, size=min(n_cells, inner.size), replace=False))
    offs = [(di, dj) for di in range(-win_i, win_i + 1) for dj in range(-win_j, win_j + 1)]
    nb = np.stack([band.shift(di, dj)[cells] for di, dj in offs], axis=1)     # (c, K)

    def cross(a_, b_):
        a_, b_ = a_ - a_.mean(0), b_ - b_.mean(0)
        return np.einsum("nc,nck->ck", b_[:, cells], a_[:, nb]) / (a_.shape[0] - 1)

    C_ref = cross(ha[N:], hb[N:])
    norm = np.sqrt((C_ref ** 2).sum(axis=1))
    out = {"sample": np.sqrt(((cross(ha[:N], hb[:N]) - C_ref) ** 2).sum(axis=1)) / norm}
    for f in files:
        fac = io.load_factor(f)
        ts = TriangularSolver(fac.L)
        E = np.zeros((fac.dim, cells.size))
        E[n + 2 * m + cells, np.arange(cells.size)] = 1.0
        B = ts.solve(ts.solve_t(E) / fac.d[:, None])                          # columns of B
        C = B[2 * m + nb, np.arange(cells.size)[:, None]]
        name = f.name[len(f"precision_{tag}_enks-"):-4]
        out[name] = np.sqrt(((C - C_ref) ** 2).sum(axis=1)) / norm
    return out, cells, io.load_npz(sdir / f"wake_{tag}.npz")["length"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", default=None, help="E-S1 run folder name under results/es1")
    p.add_argument("--r", type=int, default=2)
    p.add_argument("--window", type=int, nargs=2, default=[8, 20], help="rows, cols half-widths")
    a = p.parse_args()
    model, _ = make_model("two_jets")
    band = Band.from_model(model)
    structure_report(band, a.r, 2)
    if a.run is None:
        return
    run = common.RESULTS / "es1" / a.run
    cfg = common.json_load(run / "config.json")
    if not cfg.get("n_ref"):
        raise SystemExit("this run has no reference ensemble (n_ref = 0)")
    rows = []
    for lag in cfg["lags"]:
        rows += physics(band, run, cfg["gallery_seed"], lag, cfg["N"][0], a.r, *a.window)
        cv, length = predictive(band, run, cfg["gallery_seed"], lag, cfg["N"][0], a.r)
        for row in rows[-(len(BINS) + 1):]:
            if row["lag_h"] != lag or not cv:
                continue
            lo = float(row["length_bin"].split("-")[0].replace(">=", ""))
            hi = 1e9 if row["length_bin"].startswith(">=") else float(row["length_bin"].split("-")[1])
            sel = (length >= lo) & (length < hi)
            for name, frac in cv.items():
                row[f"cv_{name}"] = float(np.average(frac[sel], weights=band.weights[sel]))
        cov, ccells, clen = implied_covariance(band, run, cfg["gallery_seed"], lag, cfg["N"][0],
                                               *a.window)
        for row in rows[-(len(BINS) + 1):]:
            if row["lag_h"] != lag or not cov:
                continue
            lo = float(row["length_bin"].split("-")[0].replace(">=", ""))
            hi = 1e9 if row["length_bin"].startswith(">=") else float(row["length_bin"].split("-")[1])
            sel = (clen[ccells] >= lo) & (clen[ccells] < hi)
            for name, err in cov.items():
                row[f"cov_{name}"] = float(err[sel].mean()) if sel.sum() >= 5 else float("nan")
    io.write_csv(run / "verify_predecessors.csv", rows)
    cols = ["lag_h", "length_bin", "cells", "wake_cells", "peak_corr", "peak_to_traj_end",
            "peak_to_own_cell", "peak_in_wake", "peak_in_patch", "peak_in_square",
            "mass_wake", "mass_patch", "mass_square"]
    print("\n" + " ".join(f"{c:>16}" for c in cols))
    for r_ in rows:
        print(" ".join(f"{r_[c]:>16.3f}" if isinstance(r_[c], float) else f"{str(r_[c]):>16}"
                       for c in cols))
    cvc = ["lag_h", "length_bin", "cells"] + sorted(k for k in rows[0] if k.startswith("cv_"))
    if len(cvc) > 3:
        print("\nout-of-sample unexplained fraction of the h row of x_b "
              f"(trained on N = {cfg['N'][0]} members; lower is better)")
        print(" ".join(f"{c:>16}" for c in cvc))
        for r_ in rows:
            print(" ".join(f"{r_[c]:>16.4f}" if isinstance(r_.get(c), float)
                           else f"{str(r_.get(c, '')):>16}" for c in cvc))
    cov_cols = ["lag_h", "length_bin"] + sorted(k for k in rows[0] if k.startswith("cov_"))
    if len(cov_cols) > 2:
        print("\nrelative error of the implied cross-time covariance of h "
              "(against the reference ensemble; lower is better)")
        print(" ".join(f"{c.replace('cov_', '')[:22]:>23}" for c in cov_cols))
        for r_ in rows:
            print(" ".join(f"{r_[c]:>23.3f}" if isinstance(r_.get(c), float)
                           else f"{str(r_.get(c, '')):>23}" for c in cov_cols))
    print(f"\nsaved {run / 'verify_predecessors.csv'}")


if __name__ == "__main__":
    main()
