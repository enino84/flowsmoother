"""F5. Error of depth at t - lag: square, wake, and where they differ.

  python f05_error_maps.py --run pilot --lag 24 --dens 0.1
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P

p = parser(__doc__)
p.add_argument("--lag", type=int, default=None)
p.add_argument("--dens", type=float, default=0.10)
p.add_argument("--seed", type=int, default=None)
a = p.parse_args()
plt = P.setup()
run = open_run(a.run)
band, cfg = run.band, run.cfg
lag = a.lag or max(cfg["lags"])
f = run.load(f"fields_{run.tag(lag)}.npz", a.seed)
wk = run.load(f"wake_{run.tag(lag)}.npz", a.seed)
m = band.m
keys = [k for k in f if k.startswith("an_") and k.endswith(f"_{a.dens:g}")]
sq, wa = run.find_arm(keys, "square"), run.find_arm(keys, "wake")
truth = f["truth"][0][2 * m:]
err = {name: np.abs(f[k][0][2 * m:] - truth).reshape(band.nr, band.nc)
       for name, k in (("background", None), ("square", sq), ("wake", wa)) if k}
err["background"] = np.abs(f["bg_mean"][0][2 * m:] - truth).reshape(band.nr, band.nc)
diff = err["square"] - err["wake"]
elon, elat = P.edges(band)
L = wk["length"].reshape(band.nr, band.nc) if wk["length"].size == m else None
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
vmax = np.percentile(err["background"], 99)
for ax, name in zip(axes[:3], ("background", "square", "wake")):
    im = ax.pcolormesh(elon, elat, err[name], cmap=P.cmap_seq(), vmin=0, vmax=vmax, rasterized=True)
    rm = np.sqrt(np.sum(band.weights * err[name].ravel() ** 2))
    ax.set_title(f"{name}: |error| of h at t − {lag} h (rms {rm:.4f}, normalized)")
cb = fig.colorbar(im, ax=axes[:3].tolist(), fraction=0.025, pad=0.02)
cb.set_label("|error| of h (climatological sd)")
cb.outline.set_visible(False)
dmax = np.percentile(np.abs(diff), 99)
im2 = axes[3].pcolormesh(elon, elat, diff, cmap=P.cmap_div(), vmin=-dmax, vmax=dmax, rasterized=True)
axes[3].set_title("square minus wake: orange where the wake is better")
cb2 = fig.colorbar(im2, ax=[axes[3]], fraction=0.025, pad=0.02)
cb2.outline.set_visible(False)
for ax in axes:
    if L is not None:
        ax.contour(band.cell_lon()[:band.nc], band.lat_deg[band.r0:band.r1 + 1], L,
                   levels=[np.percentile(L, 75)], colors=P.INK2, linewidths=0.6)
    ax.set_ylabel("latitude")
    ax.grid(False)
axes[3].set_xlabel("longitude")
P.note(axes[3], "thin line: upper quartile of the wake length (the jets)", y=-0.22)
P.save(fig, run.figdir(), f"f05_error_maps_lag{lag:02d}")
