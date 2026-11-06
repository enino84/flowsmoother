"""Paired gain of the wake over another structure, by lag (used by F6 and F7)."""
import numpy as np
from _common import open_run

from flowsmoother import plotting as P
from flowsmoother.summary import best_arm, levels, paired, select, swap_struct


def figure(run_name, other, dens, N, name, title):
    plt = P.setup()
    run = open_run(run_name)
    rows = run.rows
    N = N or levels(rows, "N")[0]
    lags = levels(rows, "lag_h")
    metrics = [("ratio_a_h", "h at t − lag"), ("ratio_a_tot", "total at t − lag"),
               ("ratio_b_h", "h at t"), ("ratio_b_tot", "total at t")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    used = []
    for ax, (metric, label) in zip(axes, metrics):
        data = []
        for lag in lags:
            sub = select(rows, lag_h=lag, N=N, dens=float(dens))
            wake = best_arm(sub, "EnKS-MC wake")
            diff, _ = paired(sub, wake, swap_struct(wake, other), metric) if wake else (np.array([]), [])
            data.append(-100.0 * diff)
            used.append(f"{lag} h: {wake}")
        ax.axhline(0, color=P.INK2, lw=0.9)
        P.boxes(ax, data, np.arange(len(lags)), [P.METHOD["EnKS-MC wake"]] * len(lags),
                labels=[f"{lag} h" for lag in lags])
        for x, d in enumerate(data):
            if d.size:
                ax.annotate(f"{d.mean():+.1f}", (x, d.max()), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8, color=P.INK)
        ax.set_title(label)
        ax.set_xlabel("lag")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"{other} − wake, points of\nRMSE / background RMSE (×100)")
    fig.suptitle(title + f"  (N = {N}, {100 * dens:g} % of h observed, "
                 f"{len(levels(rows, 'seed'))} seeds)", x=0.06, ha="left", fontsize=11,
                 fontweight="bold", y=1.03)
    P.note(axes[0], "above zero: the wake is better. One dot per seed, same network and same "
           "perturbed observations in both arms; number = mean.  Arms: "
           + "; ".join(sorted(set(used))), y=-0.2)
    P.save(fig, run.figdir(), name)
