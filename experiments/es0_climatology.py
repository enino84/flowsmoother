"""E-S0a. Climatological record with pairs.

One uninterrupted two_jets run: spin-up, then ``n`` snapshots ``every`` days
apart; for each snapshot the states at +6, +12 and +24 h are stored too, so
the record supplies the cross-time pairs the smoother's climate prior needs.

Output: results/clim/record.npy  (n, 4, dim) float32  and record.json.
"""
import argparse
import json
import time

import common
import numpy as np

from flowsmoother.clim import build_pair_record
from flowsmoother.model import make_model

p = argparse.ArgumentParser()
p.add_argument("--spinup-days", type=float, default=20 if common.QUICK else 60)
p.add_argument("--n-snapshots", type=int, default=40 if common.QUICK else 400)
p.add_argument("--every-days", type=float, default=2.0)
p.add_argument("--offsets-h", type=float, nargs="+", default=[0, 6, 12, 24])
p.add_argument("--out", default=None)
a = p.parse_args()

out = common.results_dir("clim") if a.out is None else common.pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
model, x0 = make_model("two_jets")
t0 = time.time()
rec = build_pair_record(model, x0, a.spinup_days, a.n_snapshots, a.every_days,
                        a.offsets_h, verbose=True)
np.save(out / "record.npy", rec)
meta = dict(preset="two_jets", spinup_days=a.spinup_days, n_snapshots=a.n_snapshots,
            every_days=a.every_days, offsets_h=list(a.offsets_h),
            LMAX=int(model.grid.LMAX), dt=float(model.dt), seconds=time.time() - t0)
(out / "record.json").write_text(json.dumps(meta, indent=1))
print("saved", out / "record.npy", rec.shape, f"{time.time() - t0:.0f} s")
