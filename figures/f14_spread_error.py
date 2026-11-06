"""F14. Cycled experiment: background spread over background error per cycle,
one panel per inflation. Shows which method asks for less inflation.

  python f14_spread_error.py --run pilot --dens 0.1 --network redraw
"""
import numpy as np
from _common import open_run, parser
from _cycled import arm_style, series

from flowsmoother import plotting as P
from flowsmoother.summary import levels

p = parser(__doc__, "es2")
p.add_argument("--dens", type=float, default=None)
p.add_argument("--network", default=None)
a = p.parse_args()
plt = P.setup()
run = open_run(a.run, "es2")
rows, cfg = run.rows, run.cfg
C = cfg["cycles"]
d = a.dens or levels(rows, "dens")[-1]
nw = a.network or levels(rows, "network")[-1]
infls, arms = levels(rows, "inflation"), cfg["arms"]
x = np.arange(1, C + 1) * cfg["window_h"] / 24.0
fig, axes = plt.subplots(1, len(infls), figsize=(4.6 * len(infls), 3.6), squeeze=False, sharey=True)
for ax, infl in zip(axes[0], infls):
    for arm in arms:
        where = dict(arm=arm, inflation=float(infl), dens=float(d), network=nw)
        ratio = series(rows, "spread_bg_tot", C, **where) / series(rows, "bg_tot", C, **where)
        col, ls = arm_style(arm, arms)
        ax.plot(x, np.nanmean(ratio, axis=0), color=col, ls=ls, label=arm)
    ax.axhline(1.0, color=P.INK2, lw=0.9)
    ax.set_yscale("log")
    ax.set_title(f"inflation {infl:g}")
    ax.set_xlabel("days since the first analysis")
axes[0, 0].set_ylabel("background spread / background error")
axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
fig.suptitle(f"Is the ensemble calibrated? 1 = spread equals error ({100 * d:g} % observed, "
             f"sites {nw})", x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.03)
P.save(fig, run.figdir(), f"f14_spread_error_d{d:g}_{nw}")
