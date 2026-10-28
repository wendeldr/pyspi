"""Per-metric timing of the reviewed preset for the run-hotspot refactor.

Times one synthetic window with the Calculator's own per-metric reports.
Loading, preprocessing and output writing are outside the timing. The
synthetic AR(1) data stand in for EEG only for timing; not for validity.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pyspi.calculator import Calculator
from pyspi.data import Data


def synthetic(P, T, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((P, T))
    X = np.zeros((P, T))
    for t in range(1, T):
        X[:, t] = 0.6 * X[:, t - 1] + 0.2 * np.roll(X[:, t - 1], 1) + noise[:, t]
    return X


def time_window(X, configfile):
    calc = Calculator(dataset=Data(X, dim_order="ps"), configfile=str(configfile))
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        calc.compute()
    total = time.perf_counter() - start
    reports = {k: r.as_dict() for k, r in calc.metric_reports.items()}
    return total, {k: {"elapsed_seconds": r["elapsed_seconds"], "status": r["status"]}
                   for k, r in reports.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--P", type=int, default=118)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(ROOT / "pyspi" / "workflow" / "reviewed.yaml"))
    parser.add_argument("--label", default="measurement")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    X = synthetic(args.P, args.T, args.seed)
    runs = []
    for _ in range(args.repeats):
        total, metrics = time_window(X, args.config)
        runs.append({"total_seconds": total, "metrics": metrics})
        print(f"{args.label}: {total:.2f} s", flush=True)
    names = list(runs[0]["metrics"])
    median = {n: float(np.median([r["metrics"][n]["elapsed_seconds"] for r in runs])) for n in names}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "pyspi"], cwd=ROOT, text=True).strip())
    receipt = {
        "label": args.label, "shape": [args.P, args.T], "seed": args.seed,
        "config": os.path.relpath(args.config, ROOT), "commit": commit, "pyspi_dirty": dirty,
        "python": platform.python_version(), "numpy": np.__version__,
        "threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "total_median_seconds": float(np.median([r["total_seconds"] for r in runs])),
        "statuses": {n: runs[-1]["metrics"][n]["status"] for n in names},
        "median_seconds": median, "runs": runs,
    }
    Path(args.out).write_text(json.dumps(receipt, indent=1, sort_keys=True))
    top = sorted(median.items(), key=lambda kv: -kv[1])[:12]
    for name, seconds in top:
        print(f"  {seconds:7.3f}  {name}")


if __name__ == "__main__":
    main()
