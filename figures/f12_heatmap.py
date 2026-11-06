"""F12. Summary table as a heat map: method by density, one panel per field
and time, for one lag. Values are seed means of RMSE / background.

  python f12_heatmap.py --run full --lag 24
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P
from flowsmoother.summary import FAMILIES, levels, mean_se

p = parser(__doc__)
p.add_argument("--lag", type=int, default=None)
p.add_argument("--N", type=int, default=None)
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
rows = run.rows
N = a.N or levels(rows, "N")[0]
lag = a.lag or max(levels(rows, "lag_h"))
dens = levels(rows, "dens")
fams = list(FAMILIES)
panels = [("ratio_a_tot", "total at t − lag"), ("ratio_a_h", "h at t − lag"),
          ("ratio_b_tot", "total at t"), ("ratio_b_h", "h at t")]
fig, axes = plt.subplots(1, 4, figsize=(3.0 + 2.4 * len(dens), 0.5 * len(fams) + 1.8), sharey=True)
for ax, (metric, label) in zip(axes, panels):
    M = np.full((len(fams), len(dens)), np.nan)
    for i, fam in enumerate(fams):
        for j, d in enumerate(dens):
            _, vals = P.family_stats(rows, fam, metric, lag_h=lag, N=N, dens=float(d))
            M[i, j] = mean_se(vals.values())[0]
    ax.imshow(M, cmap=P.cmap_seq(), aspect="auto", vmin=np.nanmin(M), vmax=np.nanmax(M))
    best = np.nanargmin(M, axis=0)
    for i in range(len(fams)):
        for j in range(len(dens)):
            if np.isfinite(M[i, j]):
                dark = (M[i, j] - np.nanmin(M)) / max(np.ptp(M[np.isfinite(M)]), 1e-12) > 0.55
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark else P.INK,
                        fontweight="bold" if best[j] == i else "normal")
    ax.set_xticks(np.arange(len(dens)))
    ax.set_xticklabels([f"{100 * d:g} %" for d in dens])
    ax.set_yticks(np.arange(len(fams)))
    ax.set_yticklabels(fams)
    ax.set_title(label, fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle(f"RMSE / background, lag {lag} h, N = {N} (bold = best of the column)", x=0.02,
             ha="left", fontsize=11, fontweight="bold", y=1.04)
P.save(fig, run.figdir(), f"f12_heatmap_lag{lag:02d}")
