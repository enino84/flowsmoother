"""Shared set-up for the experiment scripts: single-threaded BLAS, paths, flags."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = pathlib.Path(os.environ.get("FLOWSMOOTHER_RESULTS", ROOT / "results"))
QUICK = os.environ.get("QUICK", "0") == "1"
if QUICK:                      # reduced runs never touch the real results
    RESULTS = RESULTS / "quick"


def results_dir(*parts):
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_id(clim_dir):
    """Short hash identifying a climatological record (for caches and configs)."""
    import hashlib

    return hashlib.md5((pathlib.Path(clim_dir) / "record.json").read_bytes()).hexdigest()[:10]


def arm_name(arm):
    if arm["kind"] == "letks":
        return f"letks-{arm['taper']}-r{arm['r']}"
    if arm["kind"] == "filter":
        return f"filter-r{arm['r']}-a{arm['alpha']:g}-{arm['prior']}"
    return f"enks-{arm['struct']}-r{arm['r']}-a{arm['alpha']:g}-{arm['prior']}"


def build_arms(cfg):
    """Every arm of a run, in the order they are computed."""
    arms = []
    for taper in cfg["letks_tapers"]:
        for r in cfg["letks_radii"]:
            arms.append(dict(kind="letks", taper=taper, r=r))
    for r in cfg["radii"]:
        for a in cfg["alphas"]:
            arms.append(dict(kind="filter", r=r, alpha=a, prior="off"))
        for a in cfg["alphas_prior"] if cfg["priors"] else []:
            arms.append(dict(kind="filter", r=r, alpha=a, prior="same"))
    for struct in cfg["structs"]:
        for r in cfg["radii"]:
            for a in cfg["alphas"]:
                arms.append(dict(kind="enks", struct=struct, r=r, alpha=a, prior="off"))
            if struct in cfg["prior_structs"]:
                for prior in cfg["priors"]:
                    for a in cfg["alphas_prior"]:
                        arms.append(dict(kind="enks", struct=struct, r=r, alpha=a, prior=prior))
    for arm in arms:
        arm["name"] = arm_name(arm)
    return arms


def json_load(path):
    import json

    return json.loads(pathlib.Path(path).read_text())


def parse_arm(name):
    """Inverse of ``arm_name``: ``enks-wake-r2-a0.03-off`` -> dict."""
    parts = name.split("-")
    if parts[0] == "letks":
        arm = dict(kind="letks", taper=parts[1], r=int(parts[2][1:]))
    elif parts[0] == "filter":
        arm = dict(kind="filter", r=int(parts[1][1:]), alpha=float(parts[2][1:]), prior=parts[3])
    elif parts[0] == "enks":
        arm = dict(kind="enks", struct=parts[1], r=int(parts[2][1:]), alpha=float(parts[3][1:]),
                   prior=parts[4])
    else:
        raise ValueError(f"cannot parse arm {name!r}")
    arm["name"] = arm_name(arm)
    return arm
