"""Step 1 of the numeric checks: compute pyspi's own outputs (PROJECT venv only).

Reads every ``audit/checks/manifest/<family>.json`` (or the families named on
the command line), builds each SPI from the audit unit catalogue exactly as
the regression harness does, feeds it the shared synthetic dataset through the
requested preprocessing path, and stores the full process-by-process matrix
under ``audit/checks/out/pyspi/<run_id>.npz``. Both orientations are therefore
available to the comparison (``S[i, j]`` and ``S[j, i]``). Bookkeeping goes to
``out/pyspi/index/<family>.json`` (one file per family, so families can be run
concurrently) and the merged ``out/pyspi/index.json`` is rebuilt after each run.
The manifest file name must equal its ``family`` and every ``run_id`` must be
unique across families (prefix it with the unit_id as below).

Manifest schema (one file per family, written by the family's check author)::

    {"family": "xcorr",
     "runs": [{"run_id": "basic.xcorr.max_sig-True|delayed_noise|1024|kernel",
               "unit_id": "basic.xcorr.max_sig-True",
               "identifier": "xcorr_max_sig-True",
               "dataset": "delayed_noise", "T": 1024,
               "path": "kernel",            # kernel | pipeline | pipeline-detrend
               "processes": null,           # optional list of process indices to keep
               "seed": 42,
               "probe": null}]}             # optional: {"module": "probes/spectral_csm.py", "function": "coherency"}

A ``probe`` run calls ``function(x, run)`` from ``audit/checks/probes/<module>``
in the project venv instead of an SPI (for infrastructure units such as the
multitaper cross-spectral matrix, or to capture an intermediate quantity); the
function returns the array to store under ``value``. ``identifier`` may then be
null.

``path``: ``kernel`` = ``Data(x, normalise=False, detrend=False)`` (the array
reaches the estimator untouched); ``pipeline`` = the Calculator default
``Data(x)`` (z-score, ddof=1); ``pipeline-detrend`` = ``Data(x, detrend=True)``
(the EEG workflow order: detrend then z-score).

Run::

    .venv/bin/python audit/checks/pyspi_outputs.py [--family F ...] [--force]
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

CHECKS = Path(__file__).resolve().parent
AUDIT = CHECKS.parent
REPO = AUDIT.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(CHECKS))

from regression import harness  # noqa: E402
import make_data  # noqa: E402

OUT = CHECKS / "out" / "pyspi"
INDEX_DIR = OUT / "index"          # one index file per family so concurrent family runs never clobber each other
INDEX = OUT / "index.json"         # merged view, rewritten from INDEX_DIR after every run


def _read_json(path, attempts=20):
    """Read a JSON file that another family's run may be replacing concurrently.

    Writers use ``_write_json`` (temp file + ``os.replace``), so a reader only ever sees a
    complete file; the retry covers the window between ``exists()`` and ``read_text()`` on an
    older checkout and any file still being written by a pre-atomic writer.
    """
    import time
    for i in range(attempts):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if i == attempts - 1:
                raise
            time.sleep(0.25)


def _write_json(path, obj, **kw):
    """Atomic replace so a concurrent reader never sees a half-written file."""
    import os
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, **kw) + "\n")
    os.replace(tmp, path)


def load_index():
    """Merge every ``out/pyspi/index/<family>.json`` into one run_id -> record dict."""
    merged = {}
    if INDEX_DIR.is_dir():
        for f in sorted(INDEX_DIR.glob("*.json")):
            merged.update(_read_json(f))
    return merged


def write_index(family, entries):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(INDEX_DIR / f"{family}.json", entries)
    _write_json(INDEX, load_index())


def load_units():
    doc = yaml.safe_load((AUDIT / "units.yaml").read_text())
    by_identifier = {}
    for u in doc["units"]:
        for ident in u["identifiers"]:
            by_identifier[ident] = (u, ident)
    return doc, by_identifier


def entry_for(identifier, by_identifier):
    u, ident = by_identifier[identifier]
    params = u["identifier_params"][ident]
    return dict(identifier=ident, module=f".statistics.{u['module']}", cls=u["pyspi_class"],
                params=params, labels=u["labels"], deps=u["deps"])


def make_pyspi_data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(f"unknown path {path!r}")


def run_probe(probe, x, run):
    import importlib.util
    path = CHECKS / probe["module"]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return np.asarray(getattr(mod, probe["function"])(x, run))


def _spec_of(run):
    """Hashable description of a run; a changed manifest entry invalidates the cached output."""
    keys = ("unit_id", "identifier", "dataset", "T", "path", "processes", "seed", "probe")
    return json.dumps({k: run.get(k) for k in keys}, sort_keys=True)


def run_one(run, by_identifier):
    ds = make_data.load(run["dataset"], run["T"])
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    harness.seed_all(int(run.get("seed", 42)))
    t0 = time.perf_counter()
    if run.get("probe"):
        S = run_probe(run["probe"], x, run)
        return S, time.perf_counter() - t0, x
    entry = entry_for(run["identifier"], by_identifier)
    data = make_pyspi_data(x, run.get("path", "kernel"))
    spi = harness.build_spi(entry)
    S = harness.compute_spi(spi, data)
    return S, time.perf_counter() - t0, data.to_numpy(squeeze=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", help="restrict to manifest family (repeatable)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    harness.ensure_jvm()
    harness.import_statistics_modules()
    _, by_identifier = load_units()
    manifests = sorted((CHECKS / "manifest").glob("*.json"))
    n_ok = n_err = n_skip = 0
    for mf in manifests:
        m = json.loads(mf.read_text())
        family = m["family"]
        if a.family and family not in a.family:
            continue
        if mf.stem != family:
            print(f"WARNING manifest {mf.name} declares family {family!r}; file stem must equal the family", file=sys.stderr)
        fam_index_path = INDEX_DIR / f"{family}.json"
        index = _read_json(fam_index_path) if fam_index_path.is_file() else {}
        wanted = {run["run_id"] for run in m["runs"]}
        index = {rid: rec for rid, rec in index.items() if rid in wanted}   # drop runs removed from the manifest
        for run in m["runs"]:
            rid = run["run_id"]
            if "|" in rid and not rid.startswith(run["unit_id"]):
                print(f"WARNING {rid}: run_id should start with its unit_id", file=sys.stderr)
            target = OUT / f"{rid}.npz"
            if target.is_file() and not a.force and rid in index and index[rid].get("status") == "ok" \
                    and index[rid].get("spec") == _spec_of(run):
                n_skip += 1
                continue
            rec = dict(run, family=family, spec=_spec_of(run))
            try:
                S, dt, xin = run_one(run, by_identifier)
                np.savez_compressed(target, value=S, input=xin)
                rec.update(status="ok", time_s=dt, shape=list(S.shape),
                           nan_count=int(np.isnan(S).sum()))
                n_ok += 1
            except Exception as exc:  # record, never abort the batch
                rec.update(status="error", error=repr(exc), traceback=traceback.format_exc()[-2000:])
                n_err += 1
                print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            index[rid] = rec
            write_index(family, index)
        write_index(family, index)
    print(f"pyspi outputs: {n_ok} computed, {n_skip} skipped (cached), {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
