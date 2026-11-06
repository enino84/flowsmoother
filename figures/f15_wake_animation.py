"""F15. Animation: the wakes follow the flow through the cycles (for talks).

Reads the gallery run of E-S2 (truth of every cycle and the wake paths of a
sample of cells). Writes a GIF.

  python f15_wake_animation.py --run pilot
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import io
from flowsmoother import plotting as P

p = parser(__doc__, "es2")
p.add_argument("--fps", type=int, default=6)
a = p.parse_args()
plt = P.setup()
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

run = open_run(a.run, "es2")
band, cfg = run.band, run.cfg
files = sorted(f for f in run.gallery.glob("gallery_enks-wake*.npz"))
if not files:
    raise SystemExit("no gallery file of a wake arm in this run")
g = io.load_npz(files[0])
with np.load(run.gallery / "start.npz") as f:
    truth = f["truth"]
elon, elat = P.edges(band)
zs = [P.vorticity(run.model, truth[k + 1])[band.r0:band.r1 + 1] for k in range(g["rho"].shape[0])]
lim = np.percentile(np.abs(zs[0]), 99)
fig, ax = plt.subplots(figsize=(11, 3.6))
mesh = ax.pcolormesh(elon, elat, zs[0], cmap=P.cmap_field(), vmin=-lim, vmax=lim, rasterized=True)
lines = [ax.plot([], [], color=P.TIME_A, lw=1.2)[0] for _ in range(g["rho"].shape[1])]
dots = ax.scatter([], [], s=8, color=P.TIME_B, zorder=3, linewidth=0)
ax.set_xlim(elon[0], elon[-1])
ax.set_ylim(elat[-1], elat[0])
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.grid(False)
title = ax.set_title("")


def frame(k):
    mesh.set_array(zs[k].ravel())
    starts = []
    for q, ln in enumerate(lines):
        lon, lat = P.rowcol_to_lonlat(band, g["rho"][k, q], g["kap"][k, q])
        ok = np.logical_and.accumulate((lat <= elat[0]) & (lat >= elat[-1]))
        ln.set_data(lon[ok], lat[ok])
        starts.append((lon[0], lat[0]))
    dots.set_offsets(np.array(starts))
    title.set_text(f"cycle {k + 1}: day {(k + 1) * cfg['window_h'] / 24:.2f}  —  wakes of the "
                   f"last {cfg['window_h']} h, mean length {g['length'][k].mean():.1f} cells")
    return [mesh, dots, title] + lines


anim = FuncAnimation(fig, frame, frames=g["rho"].shape[0], blit=False)
out = run.figdir()
out.mkdir(parents=True, exist_ok=True)
anim.save(out / "f15_wake_animation.gif", writer=PillowWriter(fps=a.fps), dpi=90)
print("saved", out / "f15_wake_animation.gif")
