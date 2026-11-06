"""E-S2. Cycled lag-1 smoother: does the cross-time wake move the total error?

Every run is one (seed, density, network, inflation, arm). The start is the
twin construction of E-S1 (noise added once, adjustment runs); from then on
the ensemble of cycle k is the analysis of cycle k-1 propagated six hours.

Each cycle
  1. propagate the analysis members t_{k-1} -> t_k                  (forecast)
  2. inflate the anomalies of the joint state [x_{k-1}, x_k]        (multiplicative)
  3. observe depth of the truth at t_k (fixed sites, or redrawn every cycle)
  4. analyse the joint state; x_k is filtered, x_{k-1} is smoothed
  5. the filtered x_k is the start of the next cycle

The prior block of the joint precision (rows of x_{k-1}) is estimated from
the analysis ensemble of the previous cycle, which is the ensemble's sample of
that posterior; the transition block (rows of x_k) is estimated from the pair,
with the wake recomputed from the mean wind of the cycle.

Outputs in results/es2/<tag>/
  config.json, metrics.csv (one row per run and cycle)
  seed<k>/start.npz, seed<k>/runs/<run>.csv, gallery_<run>.npz (gallery run)

Usage
  python es2_cycled.py --preset pilot --workers 8
  python es2_cycled.py --preset pilot --cycles 120 --tag long --seeds 0
  QUICK=1 python es2_cycled.py
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
import numpy as np

from flowsmoother import io
from flowsmoother.analysis import TriangularSolver, enks_mc, letks
from flowsmoother.clim import Record
from flowsmoother.ensemble import draw_network, make_twin, observe_h
from flowsmoother.grid import Band
from flowsmoother.metrics import LABELS, rmse, spread
from flowsmoother.model import HOUR, Propagator, make_model
from flowsmoother.smoother import JointEstimator
from flowsmoother.wake import MeanWind, wakes_for

BASE = dict(
    scale=0.05, t1_days=2.0, t2_days=2.0, obs_sigma_m=6.0, width=2, max_cells=40.0,
    sigma="insample", alpha_clim=0.1, window_h=6, skip=4, lat_north=63.0, lat_south=-60.5,
    gallery=dict(dens=0.10, network="redraw", inflation=1.02),
)
ARMS = ["letks-gc-r4", "letks-gc-r8", "filter-r2-a0.03-off", "enks-square-r2-a0.03-off",
        "enks-wake-r2-a0.03-off", "enks-patch-r2-a0.03-off"]
PRESETS = {
    "quick": dict(seeds=[0], N=10, cycles=6, skip=1, dens=[0.10], networks=["redraw"],
                  inflations=[1.02], t1_days=0.5, t2_days=0.5,
                  arms=["letks-gc-r4", "filter-r2-a0.03-off", "enks-wake-r2-a0.03-off"]),
    "pilot": dict(seeds=[0], N=20, cycles=60, dens=[0.10], networks=["redraw", "fixed"],
                  inflations=[1.00, 1.02, 1.05, 1.10], arms=ARMS),
    "full": dict(seeds=[0, 1, 2], N=20, cycles=60, dens=[0.10, 0.03],
                 networks=["redraw", "fixed"], inflations=[1.00, 1.02, 1.05, 1.10],
                 arms=ARMS + ["enks-square-r2-a1-full", "enks-wake-r2-a1-full"]),
}
_MODEL = None


def model_here():
    global _MODEL
    if _MODEL is None:
        _MODEL = make_model("two_jets")[0]
    return _MODEL


def run_name(arm, dens, network, infl):
    return f"{arm}_d{dens:g}_{network}_i{infl:g}"


def make_start(seed, cfg, outdir, workers):
    """Truth and free run for every cycle and the ensemble at t0 (shared by all runs)."""
    path = common.pathlib.Path(outdir) / f"seed{seed:02d}" / "start.npz"
    if path.exists():
        return
    rec = Record(cfg["clim_dir"])
    w = cfg["window_h"] * HOUR
    with Propagator(workers=workers) as prop:
        twin = make_twin(prop, rec.states(0.0), seed, cfg["N"], cfg["scale"],
                         cfg["t1_days"], cfg["t2_days"], horizon_h=0)
        pair = np.stack([twin.truth[0], twin.free[0]])
        traj = [pair]
        for k in range(cfg["cycles"]):
            pair = prop.run(pair, twin.t0 + k * w, twin.t0 + (k + 1) * w)
            traj.append(pair)
    traj = np.stack(traj)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, t0=twin.t0, truth=traj[:, 0], free=traj[:, 1], ens=twin.ens[0])
    io.write_json(path.with_suffix(".json"), dict(seed=seed, draws=twin.draws))


def run_one(job):
    seed, dens, network, infl, arm_name, cfg, outdir = job
    sdir = common.pathlib.Path(outdir) / f"seed{seed:02d}"
    name = run_name(arm_name, dens, network, infl)
    csv_path = sdir / "runs" / f"{name}.csv"
    if csv_path.exists():
        return f"{name}: already done"
    t_start = time.time()
    arm = common.parse_arm(arm_name)
    model = model_here()
    band = Band.from_model(model, cfg["lat_north"], cfg["lat_south"])
    rec = Record(cfg["clim_dir"])
    sd = rec.field_sd(band)
    scale = band.scale_vector(sd)
    norm = lambda x: band.extract(x) / scale                      # noqa: E731
    r_var = (cfg["obs_sigma_m"] / sd[2]) ** 2
    with np.load(sdir / "start.npz") as f:
        t0, truth, free, X = float(f["t0"]), f["truth"], f["free"], f["ens"].astype(float)
    N = X.shape[0]
    w = cfg["window_h"] * HOUR
    est = None
    if arm["kind"] != "letks":
        clim = None
        if arm["prior"] != "off":
            clim = rec.joint_anomalies(band, cfg["window_h"], sd)
        est = JointEstimator(band, clim, cfg["alpha_clim"], cfg["cache_dir"],
                             cache_tag=f"{cfg['record_id']}|lag{cfg['window_h']}|{band.r0}-{band.r1}",
                             sigma=cfg["sigma"])
    fixed = draw_network(band, dens, np.random.default_rng([seed, int(round(dens * 1e4))]))
    g = cfg["gallery"]
    gallery = (seed == cfg["gallery_seed"] and abs(dens - g["dens"]) < 1e-9
               and network == g["network"] and abs(infl - g["inflation"]) < 1e-9)
    gal = dict(length=[], rho=[], kap=[], an_mean=[], bg_mean=[])
    rows, status = [], "ok"
    for k in range(1, cfg["cycles"] + 1):
        ta, tb = t0 + (k - 1) * w, t0 + k * w
        try:
            Xf = np.stack([model.propagate(x, [ta, tb]) for x in X])
        except RuntimeError as err:                    # the model blew up: the filter diverged
            status = f"diverged at cycle {k}: {err}"
            break
        Za, Zb = norm(X), norm(Xf)
        bg_mean = Zb.mean(0)
        Za = Za.mean(0) + infl * (Za - Za.mean(0))
        Zb = bg_mean + infl * (Zb - bg_mean)
        cells = fixed if network == "fixed" else draw_network(
            band, dens, np.random.default_rng([seed, int(round(dens * 1e4)), k]))
        y = observe_h(band, truth[k], cells, cfg["obs_sigma_m"],
                      np.random.default_rng([seed, int(round(dens * 1e4)), k, 1])) / sd[2]
        rng = np.random.default_rng([seed, int(round(dens * 1e4)), k, 2])
        zt_a, zt_b = norm(truth[k - 1]), norm(truth[k])
        wake_len = wake_cnt = np.nan
        if arm["kind"] == "letks":
            A = letks(np.stack([Za, Zb]), band, cells, y, r_var, arm["r"], arm["taper"])
            Aa, Ab = A[0], A[1]
        elif arm["kind"] == "filter":
            est.set_ensemble(Za, Zb)
            fac = est.filter_factor(arm["r"], arm["alpha"], arm["prior"])
            Ab = enks_mc(Zb, fac, 2 * band.m + cells, y, r_var, rng, TriangularSolver(fac.L))
            Aa = Za
        else:
            est.set_ensemble(Za, Zb)
            fs = model.field_size
            shape = (2, model.Nlat, model.Nlon)
            means = np.stack([X.mean(0), Xf.mean(0)])
            wind = MeanWind([ta, tb], means[:, :fs].reshape(shape),
                            means[:, fs:2 * fs].reshape(shape), band.radius)
            wakes = wakes_for([arm["struct"]], band, wind, tb, w, cfg["width"], cfg["max_cells"])
            if not wakes:                              # square: wake only as a diagnostic
                wakes = wakes_for(["wake"], band, wind, tb, w, cfg["width"], cfg["max_cells"])
            est.set_wake(wakes)
            wk = next(iter(wakes.values()))
            wake_len, wake_cnt = float(wk.length.mean()), float(wk.count.mean())
            fac = est.factor(arm["struct"], arm["r"], arm["alpha"], arm["prior"])
            A = enks_mc(np.concatenate([Za, Zb], axis=1), fac, band.dim + 2 * band.m + cells,
                        y, r_var, rng, TriangularSolver(fac.L))
            Aa, Ab = A[:, :band.dim], A[:, band.dim:]
            if gallery:
                keep = slice(None, None, 29)
                gal["length"].append(wk.length)
                gal["rho"].append(wk.rho[keep])
                gal["kap"].append(wk.kap[keep])
        if not np.all(np.isfinite(Ab)):
            status = f"non-finite analysis at cycle {k}"
            break
        row = dict(seed=seed, dens=dens, network=network, inflation=infl, arm=arm_name,
                   kind=arm["kind"], N=N, cycle=k, n_obs=cells.size)
        scores = dict(bg=rmse(band, bg_mean, zt_b), an=rmse(band, Ab.mean(0), zt_b),
                      sm=rmse(band, Aa.mean(0), zt_a), prev=rmse(band, norm(X).mean(0), zt_a),
                      free=rmse(band, norm(free[k]), zt_b),
                      spread_bg=spread(band, Zb), spread_an=spread(band, Ab))
        for key, val in scores.items():
            for i, lab in enumerate(LABELS):
                row[f"{key}_{lab}"] = val[i]
        row["wake_length_mean"], row["wake_cells_mean"] = wake_len, wake_cnt
        rows.append(row)
        if gallery:
            gal["an_mean"].append(Ab.mean(0))
            gal["bg_mean"].append(bg_mean)
        X = band.insert(Xf, Ab * scale)
    for row in rows:
        row["status"] = "ok" if status == "ok" else "diverged"
    io.write_csv(csv_path, rows)
    if gallery and rows:
        io.save_npz(sdir / f"gallery_{name}.npz", t0=t0, window_h=cfg["window_h"], sd=sd,
                    traj_cells=np.arange(band.m)[::29],
                    **{k_: np.stack(v) for k_, v in gal.items() if v})
    return f"{name}: {status}, {len(rows)} cycles in {time.time() - t_start:.0f} s"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preset", default="quick" if common.QUICK else "pilot", choices=PRESETS)
    p.add_argument("--tag", default=None)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--N", type=int)
    p.add_argument("--cycles", type=int)
    p.add_argument("--skip", type=int)
    p.add_argument("--dens", type=float, nargs="+")
    p.add_argument("--networks", nargs="+", choices=["redraw", "fixed"])
    p.add_argument("--inflations", type=float, nargs="+")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--sigma", choices=["insample", "dof", "loo"])
    p.add_argument("--workers", type=int, default=common.os.cpu_count())
    p.add_argument("--clim", default=None)
    a = p.parse_args()

    cfg = dict(BASE)
    cfg.update(PRESETS[a.preset])
    for k in ("seeds", "N", "cycles", "skip", "dens", "networks", "inflations", "arms", "sigma"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    for name in cfg["arms"]:
        common.parse_arm(name)
    clim_dir = common.pathlib.Path(a.clim) if a.clim else common.RESULTS / "clim"
    if not (clim_dir / "record.npy").exists():
        raise SystemExit(f"no record in {clim_dir}: run es0_climatology.py first")
    outdir = common.results_dir("es2", a.tag or a.preset)
    cfg.update(clim_dir=str(clim_dir), record_id=common.record_id(clim_dir),
               cache_dir=str(common.results_dir("clim", "centres")),
               gallery_seed=cfg["seeds"][0], preset=a.preset, versions=io.versions())
    io.write_json(outdir / "config.json", cfg)
    t0 = time.time()
    for seed in cfg["seeds"]:
        make_start(seed, cfg, outdir, a.workers)
    print(f"starts ready ({time.time() - t0:.0f} s)", flush=True)
    jobs = [(s, d, nw, i, arm, cfg, str(outdir)) for s in cfg["seeds"] for d in cfg["dens"]
            for nw in cfg["networks"] for i in cfg["inflations"] for arm in cfg["arms"]]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for msg in ex.map(run_one, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(run_one(job), flush=True)
    rows = io.merge_csv(outdir.glob("seed*/runs/*.csv"), outdir / "metrics.csv")
    print(f"{len(rows)} rows -> {outdir / 'metrics.csv'}  ({time.time() - t0:.0f} s, "
          f"{len(jobs)} runs)")


if __name__ == "__main__":
    main()
