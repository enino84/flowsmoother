"""Files written by the experiments (section 7 of the plan) and their readers.

Rule: no figure reruns the model or the filter; everything a figure needs is
in one of these files. Arrays are float32; every run has a ``config.json``.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import platform
import tempfile

import numpy as np
import scipy.sparse as sp


def versions():
    import matplotlib
    import pyshtools
    import scipy
    import swesphere

    from . import __version__

    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, pyshtools=pyshtools.__version__,
                swesphere=getattr(swesphere, "__version__", "0.6.0"),
                matplotlib=matplotlib.__version__, flowsmoother=__version__)


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(type(o))


def save_npz(path, **arrays):
    """Atomic compressed save; float64 arrays are stored as float32."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in arrays.items():
        v = np.asarray(v)
        out[k] = v.astype(np.float32) if v.dtype == np.float64 else v
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save_factor(path, factor):
    L = factor.L.tocsr()
    save_npz(path, data=L.data, indices=L.indices, indptr=L.indptr,
             shape=np.array(L.shape), d=factor.d)


def load_factor(path):
    from .rows import Factor

    f = load_npz(path)
    L = sp.csr_matrix((f["data"].astype(float), f["indices"], f["indptr"]),
                      shape=tuple(f["shape"]))
    return Factor(L=L, d=f["d"].astype(float))


def write_csv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    """List of dicts with numbers converted where possible."""
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append({k: _num(v) for k, v in row.items()})
    return out


def _num(v):
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def merge_csv(parts, out):
    rows = []
    for p in sorted(parts):
        rows.extend(read_csv(p))
    write_csv(out, rows)
    return rows
