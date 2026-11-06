"""Command line and paths shared by the figure scripts.

No figure runs the model or the filter: each one reads files written by the
experiments (results/es1/<run>, results/es2/<run>).
"""
import argparse
import os
import pathlib
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = pathlib.Path(os.environ.get("FLOWSMOOTHER_RESULTS", ROOT / "results"))
QUICK = os.environ.get("QUICK", "0") == "1"
if QUICK:
    RESULTS = RESULTS / "quick"
sys.path.insert(0, str(ROOT / "experiments"))


def parser(doc, exp="es1"):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", default="quick" if QUICK else "pilot",
                   help=f"run folder under results/{exp}")
    return p


def open_run(name, exp="es1"):
    from flowsmoother.plotting import Run

    path = RESULTS / exp / name
    if not (path / "config.json").exists():
        raise SystemExit(f"no run at {path}")
    return Run(path)
