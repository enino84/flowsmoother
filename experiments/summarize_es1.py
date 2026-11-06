"""Tables of an E-S1 run: best arm of every family, paired differences, wakes.

  python summarize_es1.py --run pilot
"""
import argparse

import common
import numpy as np

from flowsmoother import io
from flowsmoother.summary import (FAMILIES, best_arm, by_arm, levels, mean_se, paired, select,
                                  swap_struct)

p = argparse.ArgumentParser()
p.add_argument("--run", default="quick" if common.QUICK else "pilot")
a = p.parse_args()
run = common.RESULTS / "es1" / a.run
rows = io.read_csv(run / "metrics.csv")
out = []
for lag in levels(rows, "lag_h"):
    for N in levels(rows, "N"):
        for d in levels(rows, "dens"):
            sub = select(rows, lag_h=lag, N=N, dens=float(d))
            if not sub:
                continue
            free = mean_se(r["free_b_tot"] / r["bg_b_tot"] for r in select(sub, arm=sub[0]["arm"]))
            print(f"\nlag {lag} h, N = {N}, {100 * d:g} % of h observed "
                  f"({len(levels(sub, 'seed'))} seeds); free run / background at t: {free[0]:.3f}")
            print(f"  {'family':<24}{'best arm':<30}{'t-lag tot':>14}{'t-lag h':>14}"
                  f"{'t tot':>14}{'t h':>14}")
            for fam in FAMILIES:
                arm = best_arm(sub, fam)
                if arm is None:
                    continue
                cells = []
                rec = dict(lag_h=lag, N=N, dens=d, family=fam, arm=arm)
                for m in ("ratio_a_tot", "ratio_a_h", "ratio_b_tot", "ratio_b_h"):
                    mu, se, n = mean_se(by_arm(sub, m)[arm].values())
                    rec[m], rec[m + "_se"], rec["seeds"] = mu, se, n
                    cells.append(f"{mu:.3f}±{0 if np.isnan(se) else se:.3f}")
                out.append(rec)
                print(f"  {fam:<24}{arm:<30}" + "".join(f"{c:>14}" for c in cells))
            wake = best_arm(sub, "EnKS-MC wake")
            if wake:
                print("  paired differences (negative = the wake is better), same r, ridge and prior:")
                for other in ("square", "patch"):
                    ref = swap_struct(wake, other)
                    for m in ("ratio_a_h", "ratio_a_tot", "ratio_b_h", "ratio_b_tot"):
                        diff, seeds = paired(sub, wake, ref, m)
                        if diff.size:
                            mu, se, n = mean_se(diff)
                            print(f"    wake - {other:<7}{m:<13}{mu:+.4f} ± "
                                  f"{0 if np.isnan(se) else se:.4f}  (n = {n}, "
                                  f"wake better in {int((diff < 0).sum())})")
io.write_csv(run / "summary_best.csv", out)
print("\nwakes (cells of path length; from the wake files of every seed)")
for lag in levels(rows, "lag_h"):
    L, C, cap, left = [], [], [], []
    for f in sorted(run.glob(f"seed*/wake_lag{lag:02d}_N*.npz")):
        w = io.load_npz(f)
        L.append(w["length"]); C.append(w["count"]); cap.append(w["hit_cap"]); left.append(w["left_band"])
    if L:
        L, C = np.concatenate(L), np.concatenate(C)
        print(f"  lag {lag:2d} h: length mean {L.mean():.2f}, median {np.median(L):.2f}, "
              f"p90 {np.percentile(L, 90):.2f}, max {L.max():.1f}; cells mean {C.mean():.0f}, "
              f"max {C.max()}; cut by cap {100 * np.mean(np.concatenate(cap)):.1f} %, "
              f"by band edge {100 * np.mean(np.concatenate(left)):.1f} %")
print(f"\nsaved {run / 'summary_best.csv'}")
