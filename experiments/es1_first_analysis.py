"""E-S1. First analysis of the lag smoother: wake vs square vs LETKS.

For every seed: build truth, free run and ensemble (twin construction with
adjustment runs), carry them to t0 + 24 h, and for every lag, ensemble size
and observation density run every arm on the joint state [x(t0), x(t0+lag)]
with depth observed at t0 + lag.

Arms
  letks-<box|gc>-r<R>              local ETKS, weights from the x_b block
  filter-r<R>-a<alpha>-<prior>     EnKF-MC on x_b only (no cross-time block)
  enks-<square|wake|patch>-r<R>-a<alpha>-<off|same|full>
                                   EnKS-MC; the structure names the cross-time
                                   predecessors of the h rows of x_b
                                   (wake50 = wake at half the wind speed, etc.)

Outputs in results/es1/<tag>/
  config.json, metrics.csv (one row per seed, lag, N, density, arm)
  seed<k>/metrics.csv, fields_*.npz, wake_*.npz
  gallery seed only: ensemble_*.npz, precision_*.npz, reference_*.npz,
  trajectory.npz

Usage
  python es1_first_analysis.py --preset pilot
  QUICK=1 python es1_first_analysis.py            # code check, minutes
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
import numpy as np

from flowsmoother import io
from flowsmoother.analysis import TriangularSolver, enks_mc, letks
from flowsmoother.clim import Record
from flowsmoother.ensemble import draw_network, hourly_truth, make_twin, observe_h
from flowsmoother.grid import Band
from flowsmoother.metrics import LABELS, rmse, spread
from flowsmoother.model import HOUR, Propagator
from flowsmoother.smoother import JointEstimator
from flowsmoother.wake import MeanWind, wakes_for

BASE = dict(
    scale=0.05, t1_days=2.0, t2_days=2.0, obs_sigma_m=6.0, width=2, max_cells=40.0,
    sigma="insample", alpha_clim=0.1, letks_tapers=["box", "gc"], letks_radii=[3, 4, 6, 8, 10, 14],
    structs=["square", "wake", "patch"], prior_structs=["square", "wake"], priors=["full"],
    alphas=[0.003, 0.03, 0.3], alphas_prior=[0.03, 0.3, 3.0],
    save_arms=["letks-gc-r8", "enks-square-r2-a0.003-off", "enks-wake-r2-a0.003-off",
               "enks-patch-r2-a0.003-off", "enks-square-r2-a0.3-full", "enks-wake-r2-a0.3-full"],
    gallery_density=0.10, n_ref=200, lat_north=63.0, lat_south=-60.5,
)
PRESETS = {
    "quick": dict(seeds=[0, 1], N=[10], lags=[6, 12], dens=[0.10], radii=[2],
                  t1_days=0.5, t2_days=0.5, letks_radii=[4, 8], alphas=[0.003],
                  alphas_prior=[0.3], n_ref=24),
    "pilot": dict(seeds=[0, 1, 2, 3, 4], N=[20], lags=[6, 12, 24], dens=[0.10], radii=[2]),
    "full": dict(seeds=list(range(20)), N=[20, 40], lags=[6, 12, 24],
                 dens=[0.01, 0.03, 0.10, 0.25], radii=[2, 3], priors=["same", "full"]),
}


def run_seed(args):
    seed, cfg, outdir = args
    out = common.pathlib.Path(outdir) / f"seed{seed:02d}"
    if (out / "metrics.csv").exists():
        return f"seed {seed}: already done"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec = Record(cfg["clim_dir"])
    gallery = seed == cfg["gallery_seed"]
    n_total = max(max(cfg["N"]), cfg["n_ref"] if gallery else 0)
    with Propagator(workers=cfg["model_workers"]) as prop:
        model = prop.model
        band = Band.from_model(model, cfg["lat_north"], cfg["lat_south"])
        twin = make_twin(prop, rec.states(0.0), seed, n_total, cfg["scale"],
                         cfg["t1_days"], cfg["t2_days"], horizon_h=max(cfg["lags"]))
    sd = rec.field_sd(band)
    scale = band.scale_vector(sd)
    norm = lambda x: band.extract(x) / scale                      # noqa: E731
    r_var = (cfg["obs_sigma_m"] / sd[2]) ** 2
    arms = common.build_arms(cfg)
    need_clim = any(a.get("prior", "off") != "off" for a in arms)
    truth_a, free_a, ens_a = twin.at(0)
    zt_a, zf_a = norm(truth_a), norm(free_a)
    nets = {d: draw_network(band, d, np.random.default_rng([seed, int(round(d * 1e4))]))
            for d in cfg["dens"]}
    if gallery:
        io.save_npz(out / "trajectory.npz", t0=twin.t0, hours=twin.hours,
                    truth=twin.truth, free=twin.free,
                    truth_hourly=hourly_truth(model, truth_a, twin.t0, max(cfg["lags"])),
                    ens_mean=twin.ens[:, :max(cfg["N"])].mean(axis=1), sd=sd)
    rows = []
    for lag in cfg["lags"]:
        truth_b, free_b, ens_b = twin.at(lag)
        zt_b, zf_b = norm(truth_b), norm(free_b)
        clim = rec.joint_anomalies(band, lag, sd) if need_clim else None
        est = JointEstimator(band, clim, cfg["alpha_clim"], cfg["cache_dir"],
                             cache_tag=f"{cfg['record_id']}|lag{lag}|{band.r0}-{band.r1}",
                             sigma=cfg["sigma"])
        del clim
        obs = {d: observe_h(band, truth_b, nets[d], cfg["obs_sigma_m"],
                            np.random.default_rng([seed, lag, int(round(d * 1e4)), 1])) / sd[2]
               for d in cfg["dens"]}
        if gallery and cfg["n_ref"] > 0:
            io.save_npz(out / f"reference_lag{lag:02d}.npz", Za=norm(ens_a[:cfg["n_ref"]]),
                        Zb=norm(ens_b[:cfg["n_ref"]]))
        for N in cfg["N"]:
            tag = f"lag{lag:02d}_N{N:03d}"
            Za, Zb = norm(ens_a[:N]), norm(ens_b[:N])
            Z = np.concatenate([Za, Zb], axis=1)
            est.set_ensemble(Za, Zb)
            times, u, v = twin.mean_wind(model, N)
            wakes = wakes_for(set(cfg["structs"]) | {"wake"}, band,
                              MeanWind(times, u, v, band.radius), twin.t0 + lag * HOUR,
                              lag * HOUR, cfg["width"], cfg["max_cells"])
            est.set_wake(wakes)
            wake = wakes["wake"]
            keep = slice(None) if gallery else slice(None, None, 29)
            for wname, wk in wakes.items():
                io.save_npz(out / f"{wname}_{tag}.npz", length=wk.length, count=wk.count,
                            disp=wk.disp, hit_cap=wk.hit_cap, left_band=wk.left_band,
                            rho=wk.rho[keep], kap=wk.kap[keep],
                            traj_cells=np.arange(band.m)[keep],
                            cells=wk.cells if gallery else np.zeros((0, 0), int))
            bg = dict(a=rmse(band, Za.mean(0), zt_a), b=rmse(band, Zb.mean(0), zt_b))
            fr = dict(a=rmse(band, zf_a, zt_a), b=rmse(band, zf_b, zt_b))
            sp_bg = dict(a=spread(band, Za), b=spread(band, Zb))
            fields = dict(truth=np.stack([zt_a, zt_b]), free=np.stack([zf_a, zf_b]),
                          bg_mean=np.stack([Za.mean(0), Zb.mean(0)]),
                          bg_spread=np.stack([Za.std(0, ddof=1), Zb.std(0, ddof=1)]))
            for d in cfg["dens"]:
                fields[f"obs_cells_{d:g}"] = nets[d]
                fields[f"obs_values_{d:g}"] = obs[d]
            ens_out = dict(Za=Za, Zb=Zb)
            for arm in arms:
                t_arm = time.time()
                fac = solver = None
                if arm["kind"] == "enks":
                    fac = est.factor(arm["struct"], arm["r"], arm["alpha"], arm["prior"])
                    solver = TriangularSolver(fac.L)
                elif arm["kind"] == "filter":
                    fac = est.filter_factor(arm["r"], arm["alpha"], arm["prior"])
                    solver = TriangularSolver(fac.L)
                t_fit = time.time() - t_arm
                for d in cfg["dens"]:
                    t_an = time.time()
                    cells, y = nets[d], obs[d]
                    rng = np.random.default_rng([seed, lag, N, int(round(d * 1e4)), 2])
                    if arm["kind"] == "letks":
                        A = letks(np.stack([Za, Zb]), band, cells, y, r_var, arm["r"], arm["taper"])
                        Aa, Ab = A[0], A[1]
                    elif arm["kind"] == "filter":
                        Ab = enks_mc(Zb, fac, 2 * band.m + cells, y, r_var, rng, solver)
                        Aa = Za
                    else:
                        A = enks_mc(Z, fac, band.dim + 2 * band.m + cells, y, r_var, rng, solver)
                        Aa, Ab = A[:, :band.dim], A[:, band.dim:]
                    an = dict(a=rmse(band, Aa.mean(0), zt_a), b=rmse(band, Ab.mean(0), zt_b))
                    sp_an = dict(a=spread(band, Aa), b=spread(band, Ab))
                    row = dict(seed=seed, lag_h=lag, N=N, dens=d, n_obs=cells.size,
                               arm=arm["name"], kind=arm["kind"], struct=arm.get("struct", ""),
                               r=arm["r"], alpha=arm.get("alpha", ""), prior=arm.get("prior", ""),
                               taper=arm.get("taper", ""))
                    for tk in "ab":
                        for i, lab in enumerate(LABELS):
                            row[f"bg_{tk}_{lab}"] = bg[tk][i]
                            row[f"an_{tk}_{lab}"] = an[tk][i]
                            row[f"ratio_{tk}_{lab}"] = an[tk][i] / bg[tk][i]
                            row[f"free_{tk}_{lab}"] = fr[tk][i]
                            row[f"spread_bg_{tk}_{lab}"] = sp_bg[tk][i]
                            row[f"spread_an_{tk}_{lab}"] = sp_an[tk][i]
                    row["npred_mean"] = fac.mean_predecessors() if fac is not None else ""
                    row["wake_cells_mean"] = float(wake.count.mean())
                    row["wake_length_mean"] = float(wake.length.mean())
                    row["secs_fit"] = t_fit
                    row["secs_analysis"] = time.time() - t_an
                    rows.append(row)
                    if arm["name"] in cfg["save_arms"]:
                        fields[f"an_{arm['name']}_{d:g}"] = np.stack([Aa.mean(0), Ab.mean(0)])
                        fields[f"sp_{arm['name']}_{d:g}"] = np.stack(
                            [Aa.std(0, ddof=1), Ab.std(0, ddof=1)])
                        if gallery and abs(d - cfg["gallery_density"]) < 1e-9:
                            ens_out[f"Aa_{arm['name']}"] = Aa
                            ens_out[f"Ab_{arm['name']}"] = Ab
                if gallery and fac is not None and arm["name"] in cfg["save_arms"]:
                    io.save_factor(out / f"precision_{tag}_{arm['name']}.npz", fac)
            io.save_npz(out / f"fields_{tag}.npz", **fields)
            if gallery:
                io.save_npz(out / f"ensemble_{tag}.npz", **ens_out)
            io.write_csv(out / "metrics_partial.csv", rows)
    io.write_json(out / "seed.json", dict(seed=seed, draws=twin.draws, t0=twin.t0,
                                          band=[band.r0, band.r1], sd=sd,
                                          seconds=time.time() - t_start))
    io.write_csv(out / "metrics.csv", rows)
    (out / "metrics_partial.csv").unlink(missing_ok=True)
    return f"seed {seed}: {len(rows)} rows in {time.time() - t_start:.0f} s"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preset", default="quick" if common.QUICK else "pilot", choices=PRESETS)
    p.add_argument("--tag", default=None, help="output folder name (default: the preset)")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--N", type=int, nargs="+")
    p.add_argument("--lags", type=int, nargs="+")
    p.add_argument("--dens", type=float, nargs="+")
    p.add_argument("--radii", type=int, nargs="+")
    p.add_argument("--alphas", type=float, nargs="+")
    p.add_argument("--alphas-prior", type=float, nargs="+")
    p.add_argument("--structs", nargs="+")
    p.add_argument("--priors", nargs="*")
    p.add_argument("--prior-structs", nargs="*")
    p.add_argument("--save-arms", nargs="+")
    p.add_argument("--n-ref", type=int)
    p.add_argument("--sigma", choices=["insample", "dof", "loo"],
                   help="residual variance of the rows (see flowsmoother.rows.fit_rows)")
    p.add_argument("--t1-days", type=float)
    p.add_argument("--t2-days", type=float)
    p.add_argument("--gallery-seed", type=int, default=None)
    p.add_argument("--seed-workers", type=int, default=1, help="seeds run in parallel")
    p.add_argument("--model-workers", type=int, default=None,
                   help="processes propagating members (default: all cores if one seed worker)")
    p.add_argument("--clim", default=None, help="folder with record.npy (default results/clim)")
    a = p.parse_args()

    cfg = dict(BASE)
    cfg.update(PRESETS[a.preset])
    for k in ("seeds", "N", "lags", "dens", "radii", "alphas", "alphas_prior", "structs",
              "priors", "prior_structs", "n_ref", "t1_days", "t2_days", "save_arms", "sigma"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg["gallery_seed"] = cfg["seeds"][0] if a.gallery_seed is None else a.gallery_seed
    clim_dir = common.pathlib.Path(a.clim) if a.clim else common.RESULTS / "clim"
    if not (clim_dir / "record.npy").exists():
        raise SystemExit(f"no record in {clim_dir}: run es0_climatology.py first")
    outdir = common.results_dir("es1", a.tag or a.preset)
    cfg.update(clim_dir=str(clim_dir), record_id=common.record_id(clim_dir),
               cache_dir=str(common.results_dir("clim", "centres")),
               model_workers=a.model_workers or (None if a.seed_workers == 1 else 1),
               preset=a.preset, versions=io.versions())
    if a.seed_workers > 1 and cfg["model_workers"] != 1:
        raise SystemExit("use --model-workers 1 when seeds run in parallel")
    io.write_json(outdir / "config.json", cfg)
    jobs = [(s, cfg, str(outdir)) for s in cfg["seeds"]]
    t0 = time.time()
    if a.seed_workers > 1:
        with ProcessPoolExecutor(a.seed_workers) as ex:
            for msg in ex.map(run_seed, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(run_seed(job), flush=True)
    rows = io.merge_csv(outdir.glob("seed*/metrics.csv"), outdir / "metrics.csv")
    print(f"{len(rows)} rows -> {outdir / 'metrics.csv'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
