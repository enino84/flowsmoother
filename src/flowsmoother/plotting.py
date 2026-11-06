"""Shared look of every figure, and the loaders the figure scripts use.

One colour per method, the same in every figure (validated for colour-vision
deficiencies; the lighter three always come with a direct label or legend),
one colour per time (t - lag blue, t orange), neutral inks for everything else.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
METHOD = {
    "LETKS": "#2a78d6",
    "EnKS-MC square": "#eb6834",
    "EnKS-MC wake": "#1baf7a",
    "EnKF-MC (filter)": "#eda100",
    "EnKS-MC patch": "#e87ba4",
    "EnKS-MC square + clim": "#eb6834",
    "EnKS-MC wake + clim": "#1baf7a",
    "free run": MUTED,
}
DASHED = {"EnKS-MC square + clim", "EnKS-MC wake + clim"}
TIME_A, TIME_B = "#2a78d6", "#eb6834"          # t - lag, t
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b79d", "#eb6834", "#a8401a"]


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0,
        "lines.markersize": 5, "text.color": INK, "savefig.dpi": 200, "figure.dpi": 100,
        "pdf.fonttype": 42,
    })
    return plt


def cmap_seq():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ)


def cmap_div():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div_blue_orange", DIV)


def cmap_field():
    """Neutral map for the background flow, so paths and predecessors stand out."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "neutral", ["#55534d", "#a9a79e", "#f4f3ee", "#cfcdc4", "#7d7b73"])


def save(fig, outdir, name):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    print("saved", outdir / f"{name}.png")


def note(ax, text, y=-0.02, **kw):
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color=MUTED, **kw)


# ------------------------------------------------------------------- loaders
class Run:
    """An experiment folder: config, metrics, gallery files."""

    def __init__(self, path):
        from . import io

        self.path = pathlib.Path(path)
        self.cfg = json.loads((self.path / "config.json").read_text())
        self.rows = io.read_csv(self.path / "metrics.csv") if (self.path / "metrics.csv").exists() else []
        self.gallery = self.path / f"seed{self.cfg['gallery_seed']:02d}"
        self._band = None

    @property
    def band(self):
        if self._band is None:
            from .grid import Band
            from .model import make_model

            model, _ = make_model("two_jets")
            self._band = Band.from_model(model, self.cfg["lat_north"], self.cfg["lat_south"])
            self.model = model
        return self._band

    def figdir(self):
        return self.path.parents[1] / "figures" / f"{self.path.parent.name}_{self.path.name}"

    def load(self, name, seed=None):
        from . import io

        d = self.gallery if seed is None else self.path / f"seed{seed:02d}"
        return io.load_npz(d / name)

    def tag(self, lag, N=None):
        if N is None:
            N = self.cfg["N"][0] if isinstance(self.cfg["N"], list) else self.cfg["N"]
        return f"lag{lag:02d}_N{N:03d}"

    def find_arm(self, files_or_keys, struct, prior="off"):
        """First saved arm of a structure and prior among names."""
        hits = sorted(k for k in files_or_keys if f"enks-{struct}-" in k and f"-{prior}" in k)
        if not hits:
            raise SystemExit(f"no saved arm for structure {struct!r}, prior {prior!r}")
        return hits[0]


def vorticity(model, x):
    """Relative vorticity (1/s) of a full state, (nlat, nlon)."""
    from swesphere import dynamics

    u, v, _ = model.unpack(np.asarray(x, dtype=float))
    return dynamics.relative_vorticity(u, v, model.grid)


def edges(band):
    """Cell-edge longitudes and latitudes of the band (for pcolormesh), degrees."""
    dlat, dlon = 180.0 / band.nlat, 360.0 / band.nlon
    lat = band.lat_deg[band.r0:band.r1 + 1]
    return (np.arange(band.nc + 1) - 0.5) * dlon, np.concatenate([lat + dlat / 2, [lat[-1] - dlat / 2]])


def rowcol_to_lonlat(band, rho, kap):
    return np.asarray(kap) * 360.0 / band.nlon, 90.0 - np.asarray(rho) * 180.0 / band.nlat


class Ortho:
    """Orthographic projection centred on a point (for the close-up maps)."""

    def __init__(self, lon0, lat0):
        self.l0, self.p0 = np.radians(lon0), np.radians(lat0)

    def __call__(self, lon, lat):
        lam, phi = np.radians(lon) - self.l0, np.radians(lat)
        x = np.cos(phi) * np.sin(lam)
        y = np.cos(self.p0) * np.sin(phi) - np.sin(self.p0) * np.cos(phi) * np.cos(lam)
        vis = np.sin(self.p0) * np.sin(phi) + np.cos(self.p0) * np.cos(phi) * np.cos(lam) > 0
        return np.where(vis, x, np.nan), np.where(vis, y, np.nan)

    def cell(self, band, c):
        """Projected corners of a band cell."""
        dlat, dlon = 180.0 / band.nlat, 360.0 / band.nlon
        lat, lon = band.cell_lat()[c], band.cell_lon()[c]
        lo = [lon - dlon / 2, lon + dlon / 2, lon + dlon / 2, lon - dlon / 2]
        la = [lat - dlat / 2, lat - dlat / 2, lat + dlat / 2, lat + dlat / 2]
        x, y = self(np.array(lo), np.array(la))
        return np.column_stack([x, y])

    def graticule(self, ax, step=15, **kw):
        style = dict(color=AXIS, lw=0.5, zorder=1)
        style.update(kw)
        t = np.linspace(-180, 180, 361)
        for lat in range(-75, 90, step):
            ax.plot(*self(t, np.full_like(t, lat)), **style)
        s = np.linspace(-85, 85, 171)
        for lon in range(-180, 180, step):
            ax.plot(*self(np.full_like(s, lon), s), **style)


def pick_cell(band, wake_length, kind="jet", north=True):
    """A representative cell: long wake in the jet, or a short one in quiet air."""
    lat = band.cell_lat()
    ok = (lat > 25) & (lat < 58) if north else (lat < -25) & (lat > -55)
    ok &= (band.cell_j > 20) & (band.cell_j < band.nc - 20)
    idx = np.where(ok)[0]
    target = np.percentile(wake_length[idx], 93 if kind == "jet" else 8)
    return int(idx[np.argmin(np.abs(wake_length[idx] - target))])


def boxes(ax, data, positions, colors, width=0.55, labels=None):
    """Thin box plots with every seed drawn on top (jittered deterministically)."""
    for k, (vals, x, col) in enumerate(zip(data, positions, colors)):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size >= 4:
            ax.boxplot(vals, positions=[x], widths=width, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.22, edgecolor=col, linewidth=1.2),
                       medianprops=dict(color=col, linewidth=2), whiskerprops=dict(color=col),
                       capprops=dict(color=col), manage_ticks=False)
        else:
            ax.hlines(np.median(vals), x - width / 2, x + width / 2, color=col, lw=2)
        jit = (np.arange(vals.size) / max(vals.size - 1, 1) - 0.5) * width * 0.6
        ax.scatter(x + jit, vals, s=22, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)


def family_stats(rows, family, metric, **where):
    """Per-seed values of the best arm of a family: ``(arm, {seed: value})``."""
    from .summary import best_arm, by_arm, select

    sub = select(rows, **where)
    arm = best_arm(sub, family)
    return (None, {}) if arm is None else (arm, by_arm(sub, metric)[arm])
