"""F13. Cycled experiment: analysis error per cycle, every method at its own
best inflation, with the free run as reference.

  python f13_cycled_rmse.py --run pilot --metric an_tot
"""
import numpy as np
from _common import open_run, parser
from _cycled import arm_style, best_inflation, series

from flowsmoother import plotting as P
from flowsmoother.summary import levels

p = parser(__doc__, "es2")
p.add_argument("--metric", default="an_tot", help="an_*, bg_*, sm_* with u, v, h, tot")
a = p.parse_args()
plt = P.setup()
run = open_run(a.run, "es2")
rows, cfg = run.rows, run.cfg
C, skip = cfg["cycles"], cfg["skip"]
dens, nets, arms = levels(rows, "dens"), levels(rows, "network"), cfg["arms"]
fig, axes = plt.subplots(len(dens), len(nets), figsize=(6.2 * len(nets), 3.6 * len(dens)),
                         squeeze=False, sharey=True, sharex=True)
x = np.arange(1, C + 1) * cfg["window_h"] / 24.0
for i, d in enumerate(dens):
    for j, nw in enumerate(nets):
        ax = axes[i, j]
        where = dict(dens=float(d), network=nw)
        free = series(rows, a.metric.replace(a.metric.split("_")[0], "free", 1), C,
                      arm=arms[0], inflation=float(levels(rows, "inflation")[0]), **where)
        ax.plot(x, np.nanmean(free, axis=0), color=P.METHOD["free run"], lw=1.4, label="free run")
        for arm in arms:
            infl, _ = best_inflation(rows, arm, a.metric, C, skip, **where)
            if infl is None:
                continue
            s = series(rows, a.metric, C, arm=arm, inflation=float(infl), **where)
            mu = np.nanmean(s, axis=0)
            col, ls = arm_style(arm, arms)
            if s.shape[0] > 1:
                ax.fill_between(x, np.nanmin(s, axis=0), np.nanmax(s, axis=0), color=col,
                                alpha=0.12, linewidth=0)
            stopped = "" if np.isfinite(mu[-1]) else ", stopped"
            ax.plot(x, mu, color=col, ls=ls, label=f"{arm}  (infl. {infl:g}{stopped})")
        ax.axvspan(0, skip * cfg["window_h"] / 24.0, color=P.GRID, alpha=0.5, linewidth=0)
        ax.set_yscale("log")
        ax.set_title(f"{100 * d:g} % of h observed, sites {'redrawn every cycle' if nw == 'redraw' else 'fixed'}")
        ax.legend(fontsize=7, loc="upper right")
        if i == len(dens) - 1:
            ax.set_xlabel("days since the first analysis")
        if j == 0:
            ax.set_ylabel(f"{a.metric} (normalized RMSE)")
fig.suptitle(f"Cycled smoother, N = {cfg['N']}, {C} cycles of {cfg['window_h']} h\n"
             f"{len(levels(rows, 'seed'))} seed(s); band = range over seeds; shaded = unscored",
             x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
P.save(fig, run.figdir(), f"f13_cycled_rmse_{a.metric}")
