"""F3. Where the dynamics lives in the matrix: the cross-time block.

Left: the factor L by blocks. Right: the cross-time block h(t) <- h(t - lag) for
the cells of one latitude circle in the jet, with the wake and with the square.

  python f03_sparsity.py --run pilot --lag 24
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import io
from flowsmoother import plotting as P

p = parser(__doc__)
p.add_argument("--lag", type=int, default=None)
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
band, cfg = run.band, run.cfg
lag = a.lag or max(cfg["lags"])
tag = run.tag(lag)
m, n = band.m, band.dim
files = [f.name for f in run.gallery.glob(f"precision_{tag}_*.npz")]
fac = {s: io.load_factor(run.gallery / run.find_arm(files, s)) for s in ("wake", "square")}
wk = run.load(f"wake_{tag}.npz")
row_mean = wk["length"].reshape(band.nr, band.nc).mean(axis=1)
north = np.where(band.lat_deg[band.r0:band.r1 + 1] > 0)[0]
north = north[north >= 5]                       # keep the zoom window inside the band
irow = int(north[np.argmax(row_mean[north])])

fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.35, wspace=0.16)
ax = fig.add_subplot(gs[:, 0])
Lc = fac["wake"].L.tocoo()
H, _, _ = np.histogram2d(Lc.row, Lc.col, bins=300, range=[[0, 2 * n], [0, 2 * n]])
ax.imshow(np.where(H > 0, np.log10(H + 1), np.nan), cmap=P.cmap_seq(), origin="upper",
          extent=[0, 2 * n, 2 * n, 0], interpolation="nearest", rasterized=True)
for k in range(1, 6):
    ax.axhline(k * m, color=P.AXIS, lw=0.5)
    ax.axvline(k * m, color=P.AXIS, lw=0.5)
ax.axhline(n, color=P.INK2, lw=0.9)
ax.axvline(n, color=P.INK2, lw=0.9)
ax.add_patch(plt.Rectangle((2 * m, n + 2 * m), m, m, fill=False, edgecolor=P.TIME_B, lw=1.6))
ticks = (np.arange(6) + 0.5) * m
ax.set_xticks(ticks)
ax.set_xticklabels(["u", "v", "h", "u", "v", "h"])
ax.set_yticks(ticks)
ax.set_yticklabels(["u", "v", "h", "u", "v", "h"])
ax.grid(False)
ax.text(0.25, 1.015, f"t − {lag} h", transform=ax.transAxes, ha="center", color=P.TIME_A,
        fontweight="bold")
ax.text(0.75, 1.015, "t", transform=ax.transAxes, ha="center", color=P.TIME_B, fontweight="bold")
ax.set_title(f"Factor L of the joint state ({fac['wake'].mean_predecessors():.0f} predecessors "
             "per row)", pad=18)
P.note(ax, "boxed: h at t on h at t − lag, shown on the right", y=-0.07)

rows = n + 2 * m + irow * band.nc + np.arange(band.nc)
half = 5
for k, s in enumerate(("wake", "square")):
    axz = fig.add_subplot(gs[k, 1])
    sub = fac[s].L[rows][:, 2 * m:3 * m].tocoo()
    ci, cj = band.cell_i[sub.col] - irow, band.cell_j[sub.col]
    sel = np.abs(ci) <= half
    x = (ci[sel] + half) * band.nc + cj[sel]
    axz.scatter(x, sub.row[sel], s=1.2, color=P.METHOD[f"EnKS-MC {s}"], linewidth=0, rasterized=True)
    for q in range(1, 2 * half + 1):
        axz.axvline(q * band.nc, color=P.GRID, lw=0.6)
    axz.set_xlim(0, (2 * half + 1) * band.nc)
    axz.set_ylim(band.nc, 0)
    axz.set_xticks((np.arange(2 * half + 1) + 0.5) * band.nc)
    axz.set_xticklabels([f"{d:+d}" if d else "same" for d in range(-half, half + 1)])
    axz.set_ylabel("cell along the circle")
    axz.grid(False)
    axz.set_title(f"{s}: h at t ({band.lat_deg[band.r0 + irow]:.0f}°) on h at t − {lag} h, "
                  f"{sub.nnz / band.nc:.0f} cells per row")
    if k == 1:
        axz.set_xlabel("grid row of the predecessor relative to the cell (each stripe is one latitude circle)")
fig.suptitle("The dynamics is written in the cross-time block: the wake moves the bands upstream",
             x=0.07, ha="left", fontsize=11, fontweight="bold", y=1.0)
P.save(fig, run.figdir(), f"f03_sparsity_lag{lag:02d}")
