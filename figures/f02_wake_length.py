"""F2. Map of the wake length (cells of path), one panel per lag.

  python f02_wake_length.py --run pilot
"""
import numpy as np
from _common import open_run, parser

from flowsmoother import plotting as P

a = parser(__doc__).parse_args()
plt = P.setup()
run = open_run(a.run)
band, lags = run.band, run.cfg["lags"]
wk = {lag: run.load(f"wake_{run.tag(lag)}.npz") for lag in lags}
vmax = max(np.percentile(w["length"], 99.5) for w in wk.values())
fig, axes = plt.subplots(len(lags), 1, figsize=(11, 2.7 * len(lags)), squeeze=False)
elon, elat = P.edges(band)
for ax, lag in zip(axes[:, 0], lags):
    L = wk[lag]["length"].reshape(band.nr, band.nc)
    im = ax.pcolormesh(elon, elat, L, cmap=P.cmap_seq(), vmin=0, vmax=vmax, rasterized=True)
    ax.set_ylabel("latitude")
    ax.grid(False)
    ax.set_title(f"lag {lag} h: median {np.median(L):.1f}, 90th percentile "
                 f"{np.percentile(L, 90):.1f} cells; {wk[lag]['count'].mean():.0f} cells in the "
                 f"wake on average")
axes[-1, 0].set_xlabel("longitude")
cb = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.025, pad=0.02)
cb.set_label("path length of the wake (grid cells)")
cb.outline.set_visible(False)
fig.suptitle("Wake length: the two jets and little else", x=0.06, ha="left", fontsize=11,
             fontweight="bold", y=0.995)
P.save(fig, run.figdir(), "f02_wake_length")
