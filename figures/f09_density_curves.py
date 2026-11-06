"""F9. Error against observation density, one curve per method (mean and
standard error over seeds), one panel per lag.

  python f09_density_curves.py --run full --metric ratio_a_tot
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P
from flowsmoother.summary import FAMILIES, levels, mean_se

p = parser(__doc__)
p.add_argument("--metric", default="ratio_a_tot")
p.add_argument("--N", type=int, default=None)
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
rows = run.rows
N = a.N or levels(rows, "N")[0]
lags, dens = levels(rows, "lag_h"), levels(rows, "dens")
fig, axes = plt.subplots(1, len(lags), figsize=(4.4 * len(lags), 3.7), squeeze=False, sharey=True)
for ax, lag in zip(axes[0], lags):
    for fam in FAMILIES:
        mu, se = [], []
        for d in dens:
            _, vals = P.family_stats(rows, fam, a.metric, lag_h=lag, N=N, dens=float(d))
            m_, s_, _ = mean_se(vals.values())
            mu.append(m_)
            se.append(0.0 if np.isnan(s_) else s_)
        mu, se = np.array(mu), np.array(se)
        x = 100 * np.array(dens)
        ax.fill_between(x, mu - se, mu + se, color=P.METHOD[fam], alpha=0.15, linewidth=0)
        ax.plot(x, mu, color=P.METHOD[fam], marker="o", ls="--" if fam in P.DASHED else "-",
                label=fam)
    ax.set_xscale("log")
    ax.set_xticks(100 * np.array(dens))
    ax.set_xticklabels([f"{100 * d:g}" for d in dens])
    ax.minorticks_off()
    ax.set_xlabel("h observed (% of cells)")
    ax.set_title(f"lag {lag} h")
axes[0, 0].set_ylabel(f"{a.metric.replace('ratio_a_', 't − lag, ').replace('ratio_b_', 't, ')}: "
                      "RMSE / background")
axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(f"Error against density of the network (N = {N}; band = standard error over seeds)",
             x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.02)
P.save(fig, run.figdir(), f"f09_density_curves_{a.metric}")
