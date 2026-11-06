"""F4. Cross-time correlation of depth for one cell: the reference ensemble,
and what the precision implies with the square and with the wake.

corr(h_t(cell), h_{t-lag}(.)) from the large reference ensemble is the truth
the structure should reproduce. The implied correlation comes from
B = L^-1 D^-1 L^-T, obtained with triangular solves on the saved factors.

  python f04_crosstime_correlation.py --run pilot --lag 24
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import io
from flowsmoother import plotting as P
from flowsmoother.analysis import TriangularSolver

p = parser(__doc__)
p.add_argument("--lag", type=int, default=None)
p.add_argument("--kind", default="jet", choices=["jet", "quiet"])
p.add_argument("--cell", type=int, nargs=2, default=None)
p.add_argument("--window", type=int, nargs=2, default=[7, 22])
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
band, cfg = run.band, run.cfg
lag = a.lag or max(cfg["lags"])
tag = run.tag(lag)
m, n = band.m, band.dim
wk = run.load(f"wake_{tag}.npz")
c = (a.cell[0] * band.nc + a.cell[1]) if a.cell else P.pick_cell(band, wk["length"], a.kind)
wi, wj = a.window
i0 = int(np.clip(band.cell_i[c], wi, band.nr - 1 - wi))
ii = np.arange(i0 - wi, i0 + wi + 1)
jc = int(np.rint(0.5 * (wk["kap"][c, 0] + wk["kap"][c, -1])))
jj = np.arange(jc - wj, jc + wj + 1)
win = (ii[:, None] * band.nc + (jj[None, :] % band.nc)).ravel()

ref = run.load(f"reference_lag{lag:02d}.npz")
ha, hb = ref["Za"][:, 2 * m + win].astype(float), ref["Zb"][:, 2 * m + c].astype(float)
ha = (ha - ha.mean(0)) / ha.std(0, ddof=1)
hb = (hb - hb.mean()) / hb.std(ddof=1)
maps = {f"reference ensemble (N = {len(hb)})": (hb @ ha / (len(hb) - 1))}
files = [f.name for f in run.gallery.glob(f"precision_{tag}_*.npz")]
for s in ("square", "wake"):
    fac = io.load_factor(run.gallery / run.find_arm(files, s))
    ts = TriangularSolver(fac.L)
    idx = np.concatenate([[n + 2 * m + c], 2 * m + win])
    E = np.zeros((fac.dim, idx.size))
    E[idx, np.arange(idx.size)] = 1.0
    V = ts.solve_t(E) / np.sqrt(fac.d)[:, None]
    var = (V ** 2).sum(axis=0)
    maps[f"implied by the {s}"] = (V[:, 0] @ V[:, 1:]) / np.sqrt(var[0] * var[1:])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
ext = [jj[0] - 0.5, jj[-1] + 0.5, ii[-1] + 0.5 + band.r0, ii[0] - 0.5 + band.r0]
for ax, (name, cm) in zip(axes, maps.items()):
    im = ax.imshow(cm.reshape(ii.size, jj.size), cmap=P.cmap_div(), vmin=-1, vmax=1, extent=ext,
                   origin="upper", interpolation="nearest", aspect="auto")
    ax.plot(wk["kap"][c], wk["rho"][c], color=P.INK, lw=1.2)
    ax.scatter(wk["kap"][c, 0], wk["rho"][c, 0], s=90, marker="*", color=P.TIME_B,
               edgecolor=P.INK, linewidth=0.6, zorder=5)
    k = int(np.argmax(np.abs(cm)))
    ax.scatter(jj[k % jj.size], ii[k // jj.size] + band.r0, s=70, marker="o", facecolor="none",
               edgecolor=P.INK, linewidth=1.3, zorder=6)
    ax.set_title(name)
    ax.set_xlabel("grid column")
    ax.grid(False)
axes[0].set_ylabel("grid row")
axes[0].scatter([], [], s=60, marker="*", color=P.TIME_B, edgecolor=P.INK, label="cell at t")
axes[0].scatter([], [], s=50, marker="o", facecolor="none", edgecolor=P.INK, label="largest |correlation|")
axes[0].plot([], [], color=P.INK, lw=1.2, label="backward path")
axes[0].legend(loc="lower left", fontsize=7.5, ncol=3, bbox_to_anchor=(0, -0.42))
cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.015)
cb.set_label(f"correlation with h at t − {lag} h")
cb.outline.set_visible(False)
fig.suptitle(f"Cross-time correlation of depth for a cell at {band.cell_lat()[c]:.0f}°, lag {lag} h",
             x=0.06, ha="left", fontsize=11, fontweight="bold", y=1.04)
P.save(fig, run.figdir(), f"f04_crosstime_correlation_lag{lag:02d}_{a.kind}")
