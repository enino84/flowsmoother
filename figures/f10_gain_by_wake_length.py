"""F10. Does the gain sit where the flow is? Error reduction of the wake over
the square per cell, grouped by the local wake length.

For every seed: mean over the cells of a length class of
(squared error with the square − squared error with the wake) of h at t − lag,
relative to the mean background squared error. One dot per seed.

  python f10_gain_by_wake_length.py --run pilot --dens 0.1
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import io
from flowsmoother import plotting as P
from flowsmoother.summary import levels

p = parser(__doc__)
p.add_argument("--dens", type=float, default=0.10)
p.add_argument("--N", type=int, default=None)
p.add_argument("--other", default="square", choices=["square", "patch"])
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
band = run.band
m = band.m
N = a.N or levels(run.rows, "N")[0]
lags = levels(run.rows, "lag_h")
bins = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 1e9)]
labels = ["< 1", "1–3", "3–6", "6–10", "> 10"]
fig, axes = plt.subplots(1, len(lags), figsize=(4.3 * len(lags), 3.6), squeeze=False, sharey=True)
for ax, lag in zip(axes[0], lags):
    data = [[] for _ in bins]
    for sdir in sorted(run.path.glob("seed*")):
        tag = f"lag{lag:02d}_N{N:03d}"
        if not (sdir / f"fields_{tag}.npz").exists():
            continue
        f, wk = io.load_npz(sdir / f"fields_{tag}.npz"), io.load_npz(sdir / f"wake_{tag}.npz")
        keys = [k for k in f if k.startswith("an_") and k.endswith(f"_{a.dens:g}")]
        truth = f["truth"][0][2 * m:]
        e_o = (f[run.find_arm(keys, a.other)][0][2 * m:] - truth) ** 2
        e_w = (f[run.find_arm(keys, "wake")][0][2 * m:] - truth) ** 2
        e_b = np.sum(band.weights * (f["bg_mean"][0][2 * m:] - truth) ** 2) / band.weights.mean() / m
        for k, (lo, hi) in enumerate(bins):
            sel = (wk["length"] >= lo) & (wk["length"] < hi)
            if sel.sum() >= 20:
                data[k].append(100.0 * np.average(e_o[sel] - e_w[sel], weights=band.weights[sel]) / e_b)
    ax.axhline(0, color=P.INK2, lw=0.9)
    P.boxes(ax, data, np.arange(len(bins)), [P.METHOD["EnKS-MC wake"]] * len(bins), labels=labels)
    ax.set_xlabel("wake length at the cell (grid cells)")
    ax.set_title(f"lag {lag} h")
    ax.grid(axis="x", visible=False)
axes[0, 0].set_ylabel(f"squared error of h at t − lag,\n{a.other} − wake (% of background)")
fig.suptitle(f"Where the wake helps: gain by local wake length (N = {N}, {100 * a.dens:g} % observed)",
             x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.03)
P.note(axes[0, 0], "above zero: the wake is better there. One dot per seed.", y=-0.2)
P.save(fig, run.figdir(), f"f10_gain_by_wake_length_{a.other}")
