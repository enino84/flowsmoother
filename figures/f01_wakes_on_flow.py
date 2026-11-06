"""F1. The flow at t with the wakes of a sample of cells, one panel per lag.

  python f01_wakes_on_flow.py --run pilot
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P

a = parser(__doc__).parse_args()
plt = P.setup()
run = open_run(a.run)
band, cfg = run.band, run.cfg
traj = run.load("trajectory.npz")
lags = cfg["lags"]
fig, axes = plt.subplots(len(lags), 1, figsize=(11, 2.9 * len(lags)), squeeze=False)
elon, elat = P.edges(band)
for ax, lag in zip(axes[:, 0], lags):
    wk = run.load(f"wake_{run.tag(lag)}.npz")
    kt = int(np.where(traj["hours"] == lag)[0][0])
    z = P.vorticity(run.model, traj["truth"][kt])[band.r0:band.r1 + 1]
    lim = np.percentile(np.abs(z), 99)
    ax.pcolormesh(elon, elat, z, cmap=P.cmap_field(), vmin=-lim, vmax=lim, rasterized=True)
    pick = [c for c in range(band.m) if band.cell_i[c] % 5 == 2 and band.cell_j[c] % 6 == 3]
    for c in pick:
        lon, lat = P.rowcol_to_lonlat(band, wk["rho"][c], wk["kap"][c])
        ok = (lat <= elat[0]) & (lat >= elat[-1])
        ok = np.logical_and.accumulate(ok)
        ax.plot(lon[ok], lat[ok], color=P.TIME_A, lw=1.3, solid_capstyle="round")
        ax.scatter(lon[0], lat[0], s=9, color=P.TIME_B, zorder=3, linewidth=0)
    ax.set_xlim(elon[0], elon[-1])
    ax.set_ylim(elat[-1], elat[0])
    ax.set_ylabel("latitude")
    ax.grid(False)
    ax.set_title(f"lag {lag} h: mean path length {wk['length'].mean():.1f} cells, "
                 f"longest {wk['length'].max():.0f}")
axes[-1, 0].set_xlabel("longitude")
axes[0, 0].plot([], [], color=P.TIME_A, lw=1.3, label="backward path of the ensemble-mean wind")
axes[0, 0].scatter([], [], s=12, color=P.TIME_B, label="cell at t")
axes[0, 0].legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.32))
fig.suptitle("The flow draws the structure: long, curved wakes in the jets, a point elsewhere",
             x=0.125, ha="left", fontsize=11, fontweight="bold", y=1.0)
fig.tight_layout()
P.save(fig, run.figdir(), "f01_wakes_on_flow")
