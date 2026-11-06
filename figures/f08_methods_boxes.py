"""F8. Every method, best configuration of each family: boxes over seeds.

  python f08_methods_boxes.py --run pilot --dens 0.1
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P
from flowsmoother.summary import FAMILIES, levels

p = parser(__doc__)
p.add_argument("--dens", type=float, default=0.10)
p.add_argument("--N", type=int, default=None)
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
rows = run.rows
N = a.N or levels(rows, "N")[0]
lags = levels(rows, "lag_h")
metrics = [("ratio_a_tot", "total at t − lag"), ("ratio_a_h", "h at t − lag"),
           ("ratio_b_tot", "total at t"), ("ratio_b_h", "h at t")]
fams = list(FAMILIES)
fig, axes = plt.subplots(len(metrics), len(lags), figsize=(4.3 * len(lags), 2.5 * len(metrics)),
                         squeeze=False, sharey="row")
for i, (metric, label) in enumerate(metrics):
    for j, lag in enumerate(lags):
        ax = axes[i, j]
        data, cols = [], []
        for fam in fams:
            arm, vals = P.family_stats(rows, fam, metric, lag_h=lag, N=N, dens=float(a.dens))
            data.append(list(vals.values()))
            cols.append(P.METHOD[fam])
        P.boxes(ax, data, np.arange(len(fams)), cols)
        for x, fam in enumerate(fams):
            if fam in P.DASHED and data[x]:
                ax.scatter([x], [np.min(data[x])], marker="_", s=0)
        ax.set_xticks(np.arange(len(fams)))
        ax.set_xticklabels([f.replace("EnKS-MC ", "").replace("EnKF-MC ", "").replace(" + clim", "\n+ clim")
                            for f in fams] if i == len(metrics) - 1 else [""] * len(fams),
                           rotation=0, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.set_title(f"lag {lag} h")
        if j == 0:
            ax.set_ylabel(f"{label}\nRMSE / background")
fig.suptitle(f"Analysis error by method (N = {N}, {100 * a.dens:g} % of h observed, "
             f"{len(levels(rows, 'seed'))} seeds; best configuration of each family)",
             x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.0)
fig.tight_layout()
P.save(fig, run.figdir(), "f08_methods_boxes")
