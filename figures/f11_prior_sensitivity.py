"""F11. Sensitivity to the ridge and to the climate prior: error against alpha
for the three variants of the prior (off, same-time only, whole row).

  python f11_prior_sensitivity.py --run pilot --lag 24 --dens 0.1
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P
from flowsmoother.summary import levels, mean_se, select

p = parser(__doc__)
p.add_argument("--lag", type=int, default=None)
p.add_argument("--dens", type=float, default=0.10)
p.add_argument("--N", type=int, default=None)
p.add_argument("--r", type=int, default=None)
p.add_argument("--metric", default="ratio_a_tot")
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
rows = run.rows
N = a.N or levels(rows, "N")[0]
lag = a.lag or max(levels(rows, "lag_h"))
r = a.r or run.cfg["radii"][0]
styles = {"off": ("-", "o", "no prior"), "same": (":", "s", "climate on the same-time part"),
          "full": ("--", "D", "climate on the whole row")}
structs = [s for s in ("square", "wake") if select(rows, kind="enks", struct=s)]
fig, axes = plt.subplots(1, len(structs), figsize=(5.2 * len(structs), 3.7), squeeze=False, sharey=True)
for ax, s in zip(axes[0], structs):
    col = P.METHOD[f"EnKS-MC {s}"]
    for prior, (ls, mk, lab) in styles.items():
        sub = select(rows, kind="enks", struct=s, prior=prior, lag_h=lag, N=N, dens=float(a.dens), r=r)
        alphas = levels(sub, "alpha")
        if not alphas:
            continue
        stats = [mean_se(x[a.metric] for x in select(sub, alpha=float(al))) for al in alphas]
        mu = np.array([q[0] for q in stats])
        se = np.nan_to_num(np.array([q[1] for q in stats]))
        ax.fill_between(alphas, mu - se, mu + se, color=col, alpha=0.12, linewidth=0)
        ax.plot(alphas, mu, color=col, ls=ls, marker=mk, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("ridge alpha (relative)")
    ax.set_title(f"cross-time {s}, r = {r}")
axes[0, 0].set_ylabel(f"{a.metric}: RMSE / background")
axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(f"Ridge and climate prior (lag {lag} h, N = {N}, {100 * a.dens:g} % observed; "
             "band = standard error)", x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.03)
P.save(fig, run.figdir(), f"f11_prior_sensitivity_lag{lag:02d}")
