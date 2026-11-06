"""E-S0b. How long must the adjustment run of the ensemble be?

Builds truth, background and ensemble as in E-S1 and follows the ensemble
through its adjustment run (T2), measuring at several times:

  spread_*      ensemble spread of u, v (m/s) and h (m) over the band
  growth_h      spread of h relative to its initial value
  imbalance     rms divergence / rms vorticity of the member perturbations
                (gravity-wave noise shows up here and decays as it radiates)
  err_h         error of the ensemble mean in h (m); ratio = spread / error
  corr_change   1 - pattern correlation between the local correlation maps of h
                at this time and at the previous one (0 = structure stopped changing)

Pick T2 where imbalance and corr_change have flattened and growth is still
well inside the linear regime (below about 2).

Output: results/spinup/spinup_check.csv
"""
import argparse

import common
import numpy as np
from swesphere import dynamics

from flowsmoother import io
from flowsmoother.clim import Record
from flowsmoother.grid import Band
from flowsmoother.model import DAY, Propagator

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--N", type=int, default=8 if common.QUICK else 20)
p.add_argument("--days", type=float, nargs="+",
               default=[0, 0.25, 0.5, 1] if common.QUICK else [0, 0.25, 0.5, 1, 2, 4, 8])
p.add_argument("--t1-days", type=float, default=0.5 if common.QUICK else 2.0)
p.add_argument("--scale", type=float, default=0.05)
p.add_argument("--workers", type=int, default=None)
a = p.parse_args()

rec = Record(common.RESULTS / "clim")
S = rec.states(0.0)
rng = np.random.default_rng(a.seed)
n = len(S)
st = lambda i: np.asarray(S[int(i)], dtype=float)                   # noqa: E731
k = rng.integers(n)
pq = rng.choice(n, 2, replace=False)
ij = [rng.choice(n, 2, replace=False) for _ in range(a.N)]
rows = []
with Propagator(workers=a.workers) as prop:
    model = prop.model
    band = Band.from_model(model)
    grid = model.grid
    truth = st(k)
    bg = truth + a.scale * (st(pq[0]) - st(pq[1]))
    truth, bg = prop.run(np.stack([truth, bg]), 0.0, a.t1_days * DAY)
    ens = np.stack([bg + a.scale * (st(i) - st(j)) for i, j in ij])
    allst = np.concatenate([[truth], ens])
    t = a.t1_days * DAY
    sample = rng.choice(band.m, 300, replace=False)
    nb = band.square(4)[sample]
    prev_map, spread0 = None, None
    rows_band = slice(band.r0, band.r1 + 1)
    wlat = np.cos(np.radians(band.lat_deg[rows_band]))[:, None]
    for d in a.days:
        allst = prop.run(allst, t, a.t1_days * DAY + d * DAY)
        t = a.t1_days * DAY + d * DAY
        Z = band.extract(allst[1:])
        zt = band.extract(allst[0])
        var = Z.var(axis=0, ddof=1)
        sp = [np.sqrt(np.sum(band.weights * var[f * band.m:(f + 1) * band.m])) for f in range(3)]
        spread0 = spread0 or sp[2]
        err_h = np.sqrt(np.sum(band.weights * (Z.mean(0) - zt)[2 * band.m:] ** 2))
        div2 = vor2 = 0.0
        mean = allst[1:].mean(axis=0)
        for x in allst[1:]:
            u, v, _ = model.unpack(x - mean)
            div2 += np.mean(wlat * dynamics.divergence(u, v, grid)[rows_band] ** 2)
            vor2 += np.mean(wlat * dynamics.relative_vorticity(u, v, grid)[rows_band] ** 2)
        h = Z[:, 2 * band.m:]
        hs = (h - h.mean(0)) / h.std(0, ddof=1)
        cmap = np.einsum("nc,nck->ck", hs[:, sample], hs[:, np.where(nb >= 0, nb, 0)]) / (a.N - 1)
        cmap = np.where(nb >= 0, cmap, 0.0).ravel()
        change = np.nan if prev_map is None else 1.0 - np.corrcoef(cmap, prev_map)[0, 1]
        prev_map = cmap
        rows.append(dict(days=d, spread_u=sp[0], spread_v=sp[1], spread_h=sp[2],
                         growth_h=sp[2] / spread0, imbalance=np.sqrt(div2 / vor2),
                         err_h=err_h, spread_over_err=sp[2] / err_h, corr_change=change))
        print(" ".join(f"{k_}={v_:.4g}" for k_, v_ in rows[-1].items()), flush=True)
out = common.results_dir("spinup")
io.write_csv(out / "spinup_check.csv", rows)
io.write_json(out / "config.json", dict(vars(a), versions=io.versions()))
print("saved", out / "spinup_check.csv")
