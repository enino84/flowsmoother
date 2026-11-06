"""F0. One particle, its journey, and its row of the Cholesky factor.

A  the journey: the flow at t, the backward path of the parcel that arrives at
   the chosen cell, the wake (cross-time predecessors, at t - lag) and the
   local square (same-time predecessors, at t)
B  the row: where that row sits in L, and the row itself opened into its six
   blocks (u, v, h at each time), zoomed around the cell
C  the weights: the cross-time coefficients of the row put back on the map

  python f00_particle_row.py --run pilot --lag 24
  python f00_particle_row.py --run pilot --lag 24 --kind quiet
  python f00_particle_row.py --run pilot --lag 24 --struct square     (the contrast)
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import io
from flowsmoother import plotting as P

p = parser(__doc__)
p.add_argument("--lag", type=int, default=None)
p.add_argument("--kind", default="jet", choices=["jet", "quiet"])
p.add_argument("--struct", default="wake", help="structure whose row is drawn")
p.add_argument("--cell", type=int, nargs=2, default=None, help="band row and column")
p.add_argument("--south", action="store_true")
a = p.parse_args()
plt = P.setup()
from matplotlib.collections import PolyCollection  # noqa: E402

run = open_run(a.run)
band, cfg = run.band, run.cfg
lag = a.lag or max(cfg["lags"])
tag = run.tag(lag)
m, n = band.m, band.dim
wk = run.load(f"wake_{tag}.npz")
c = (a.cell[0] * band.nc + a.cell[1]) if a.cell else P.pick_cell(band, wk["length"], a.kind,
                                                                not a.south)
files = [f.name for f in run.gallery.glob(f"precision_{tag}_*.npz")]
fac = io.load_factor(run.gallery / run.find_arm(files, a.struct))
traj = run.load("trajectory.npz")
kt = int(np.where(traj["hours"] == lag)[0][0])
zeta_b = P.vorticity(run.model, traj["truth"][kt])
zeta_a = P.vorticity(run.model, traj["truth"][0])

row = fac.L[n + 2 * m + c].tocoo()
cols, vals = row.col, -row.data
keep = cols != n + 2 * m + c
cols, vals = cols[keep], vals[keep]
time_b = cols >= n
fld = (cols % n) // m
cell = cols % m

lon0, lat0 = band.cell_lon()[c], band.cell_lat()[c]
tlon, tlat = P.rowcol_to_lonlat(band, wk["rho"][c], wk["kap"][c])
proj = P.Ortho(0.5 * (lon0 + tlon[-1]), 0.5 * (lat0 + tlat[-1]))

fig = plt.figure(figsize=(13.5, 7.6))
gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 0.08, 1, 1], height_ratios=[1.25, 0.42, 0.95],
                      hspace=0.5, wspace=0.28)

# ------------------------------------------------------------ A: the journey
ax = fig.add_subplot(gs[:, :3])
elon, elat = P.edges(band)
LON, LAT = np.meshgrid(elon, elat)
X, Y = proj(LON, LAT)
zb = zeta_b[band.r0:band.r1 + 1]
lim = np.percentile(np.abs(zb), 99)
Xm, Ym = np.ma.masked_invalid(X), np.ma.masked_invalid(Y)
bad = Xm.mask | Ym.mask
cellmask = bad[:-1, :-1] | bad[1:, :-1] | bad[:-1, 1:] | bad[1:, 1:]
ax.pcolormesh(np.where(bad, 0.0, X), np.where(bad, 0.0, Y), np.ma.array(zb, mask=cellmask),
              cmap=P.cmap_field(), vmin=-lim, vmax=lim, shading="flat", zorder=0, rasterized=True)
proj.graticule(ax)
cx, cy = proj(*np.meshgrid(band.cell_lon()[:band.nc], band.lat_deg[band.r0:band.r1 + 1]))
ax.contour(cx, cy, zeta_a[band.r0:band.r1 + 1], levels=np.array([-0.6, -0.3, 0.3, 0.6]) * lim,
           colors=P.INK2, linewidths=0.5, linestyles=":", zorder=2)
cross_cells = np.unique(cell[~time_b])
same_cells = np.unique(cell[time_b])
ax.add_collection(PolyCollection([proj.cell(band, k) for k in cross_cells], facecolor=P.TIME_A,
                                 edgecolor=P.SURFACE, linewidth=0.4, alpha=0.55, zorder=3))
ax.add_collection(PolyCollection([proj.cell(band, k) for k in same_cells], facecolor="none",
                                 edgecolor=P.TIME_B, linewidth=1.3, zorder=5))
px, py = proj(tlon, tlat)
ax.plot(px, py, color=P.INK, lw=1.6, zorder=6)
per_hour = max(1, int(round((len(px) - 1) / lag)))
hx, hy = px[::per_hour], py[::per_hour]
alphas = np.linspace(1.0, 0.25, len(hx))
for k in range(len(hx)):
    ax.scatter(hx[k], hy[k], s=26, color=P.INK, alpha=float(alphas[k]), edgecolor=P.SURFACE,
               linewidth=0.6, zorder=7)
x0, y0 = proj(np.array([lon0]), np.array([lat0]))
ax.scatter(x0, y0, s=150, marker="*", color=P.TIME_B, edgecolor=P.INK, linewidth=0.8, zorder=8)
ax.annotate("cell at t", (x0[0], y0[0]), xytext=(10, 12), textcoords="offset points",
            fontsize=9, color=P.INK, fontweight="bold")
ax.annotate(f"where the air was\n{lag} h earlier", (px[-1], py[-1]), xytext=(-14, -30),
            textcoords="offset points", fontsize=9, color=P.INK, ha="right",
            arrowprops=dict(arrowstyle="-", color=P.INK2, lw=0.8))
allx = np.concatenate([px, [x0[0]]])
ally = np.concatenate([py, [y0[0]]])
half = max(np.ptp(allx), np.ptp(ally)) / 2 + 0.16
ax.set_xlim(allx.mean() - half, allx.mean() + half)
ax.set_ylim(ally.mean() - half, ally.mean() + half)
ax.set_aspect("equal")
ax.axis("off")
names = {"wake": "the wake", "square": "a square", "patch": "an isotropic patch"}
ax.set_title(f"A  The journey: a parcel arriving at {lat0:.0f}°N, {lon0:.0f}°E over {lag} h",
             loc="left")
ax.scatter([], [], s=90, marker="s", color=P.TIME_A, alpha=0.55,
           label=f"cross-time predecessors at t − {lag} h ({names.get(a.struct, a.struct)}): "
                 f"{cross_cells.size} cells × u, v, h")
ax.scatter([], [], s=90, marker="s", facecolor="none", edgecolor=P.TIME_B, linewidth=1.3,
           label=f"same-time predecessors at t (square r = {cfg['radii'][0]})")
ax.plot([], [], color=P.INK, lw=1.6, marker="o", ms=4, label="backward path of the mean wind, a dot per hour")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.09), fontsize=8.5, handletextpad=0.6)

# ---------------------------------------------------------------- B: the row
axp = fig.add_subplot(gs[0, 4:])
Lc = fac.L.tocoo()
nb = 240
H, _, _ = np.histogram2d(Lc.row, Lc.col, bins=nb, range=[[0, 2 * n], [0, 2 * n]])
axp.imshow(np.where(H > 0, np.log10(H + 1), np.nan), cmap=P.cmap_seq(), origin="upper",
           extent=[0, 2 * n, 2 * n, 0], interpolation="nearest", rasterized=True)
for k in range(1, 6):
    axp.axhline(k * m, color=P.AXIS, lw=0.5)
    axp.axvline(k * m, color=P.AXIS, lw=0.5)
axp.axhline(n, color=P.INK2, lw=0.9)
axp.axvline(n, color=P.INK2, lw=0.9)
axp.axhline(n + 2 * m + c, color=P.TIME_B, lw=1.4)
ticks = (np.arange(6) + 0.5) * m
labs = ["u", "v", "h", "u", "v", "h"]
axp.set_xticks(ticks)
axp.set_xticklabels(labs)
axp.set_yticks(ticks)
axp.set_yticklabels(labs)
axp.grid(False)
axp.text(0.25, 1.02, f"t − {lag} h", transform=axp.transAxes, ha="center", color=P.TIME_A,
         fontsize=8.5, fontweight="bold")
axp.text(0.75, 1.02, "t", transform=axp.transAxes, ha="center", color=P.TIME_B, fontsize=8.5,
         fontweight="bold")
axp.set_title("B  The row in the factor L", loc="left", pad=16)
axp.annotate("this row", (2 * n, n + 2 * m + c), xytext=(6, 0), textcoords="offset points",
             va="center", fontsize=8, color=P.TIME_B, annotation_clip=False)

axs = fig.add_subplot(gs[1, 4:])
span = 5 * band.nc + band.nc // 2
for b in range(6):
    tb, f = b >= 3, b % 3
    sel = (time_b == tb) & (fld == f)
    off = cell[sel] - c
    off = off[np.abs(off) <= span]
    axs.vlines(b * (2 * span + 60) + off, 0, 1, color=P.TIME_B if tb else P.TIME_A, lw=0.9)
    axs.plot([b * (2 * span + 60)] * 2, [-0.18, -0.02], color=P.INK2, lw=0.8)
    axs.text(b * (2 * span + 60), 1.12, f"{'uvh'[f]}", ha="center", fontsize=9,
             color=P.TIME_B if tb else P.TIME_A, fontweight="bold")
    axs.text(b * (2 * span + 60), -0.5, f"{int(sel.sum())}", ha="center", fontsize=7.5, color=P.INK2)
axs.set_xlim(-span - 30, 5 * (2 * span + 60) + span + 30)
axs.set_ylim(-0.7, 1.9)
axs.axis("off")
axs.text(0.0, 1.05, "the same row, block by block, zoomed on ± 5 grid rows around the cell "
         "(tick below = the cell itself; number = predecessors)", transform=axs.transAxes,
         fontsize=7.5, color=P.MUTED, va="bottom")

# ------------------------------------------------------------ C: the weights
vmax = np.percentile(np.abs(vals[~time_b]), 99) if (~time_b).any() else 1.0
i0, j0 = band.cell_i[c], band.cell_j[c]
ci, cj = band.cell_i[cross_cells], (band.cell_j[cross_cells] - j0 + band.nc // 2) % band.nc - band.nc // 2
i_lo, i_hi = min(ci.min(), i0) - 1, max(ci.max(), i0) + 1
j_lo, j_hi = min(cj.min(), 0) - 1, max(cj.max(), 0) + 1
sub = gs[2, 4:].subgridspec(1, 3, wspace=0.08)
for f in range(3):
    axc = fig.add_subplot(sub[0, f])
    img = np.full((i_hi - i_lo + 1, j_hi - j_lo + 1), np.nan)
    sel = (~time_b) & (fld == f)
    ii = band.cell_i[cell[sel]] - i_lo
    jj = (band.cell_j[cell[sel]] - j0 + band.nc // 2) % band.nc - band.nc // 2 - j_lo
    img[ii, jj] = vals[sel]
    im = axc.imshow(img, cmap=P.cmap_div(), vmin=-vmax, vmax=vmax, origin="upper",
                    extent=[j_lo - 0.5, j_hi + 0.5, i_hi + 0.5, i_lo - 0.5], interpolation="nearest")
    axc.plot(wk["kap"][c] - j0, wk["rho"][c] - band.r0, color=P.INK, lw=1.0)
    axc.scatter([0], [i0], s=60, marker="*", color=P.TIME_B, edgecolor=P.INK, linewidth=0.6, zorder=5)
    axc.set_xticks([])
    axc.set_yticks([])
    axc.grid(False)
    for s_ in axc.spines.values():
        s_.set_visible(True)
        s_.set_color(P.AXIS)
    axc.set_title(f"{'uvh'[f]} at t − {lag} h", fontsize=8.5, fontweight="normal", loc="center",
                  color=P.TIME_A)
    if f == 0:
        axc.text(0.0, 1.42, "C  The weights of the cross-time predecessors", transform=axc.transAxes,
                 fontsize=10, fontweight="bold", va="bottom")
cb = fig.colorbar(im, ax=fig.axes[-3:], orientation="horizontal", fraction=0.07, pad=0.06, aspect=40)
cb.set_label("coefficient of the row (normalized units)", fontsize=8)
cb.outline.set_visible(False)
P.save(fig, run.figdir(), f"f00_particle_row_lag{lag:02d}_{a.kind}_{a.struct}")
