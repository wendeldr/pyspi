"""Compare basic calculations with their committed pre-refactor versions.

Run: .venv/bin/python -B tests/benchmark_basic.py
The old code exists only in the benchmark's isolated module. Production has
one implementation. Each measurement uses fresh Data; paired variants share
that Data, as in Calculator. No fitting or numerical definitions are changed.
"""

import argparse
import contextlib
import datetime
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy import stats
from threadpoolctl import threadpool_info, threadpool_limits

from pyspi.data import Data
from pyspi.statistics import basic


def load_baseline(revision):
    revision = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:pyspi/statistics/basic.py"], cwd=ROOT, text=True)
    module = types.ModuleType("basic_benchmark_baseline")
    exec(compile(source, "<committed basic baseline>", "exec"), module.__dict__)
    return module, revision


def run(module, specs, data):
    return np.stack([getattr(module, name)(**kwargs).multivariate(data) for name, kwargs in specs])


def unique_scipy_kendall(specs, data):
    """Benchmark the simpler alternative: unique pairs, shared signed result."""
    x = data.to_numpy(realisation=0)
    matrix = np.full((len(x), len(x)), np.nan)
    for j in range(len(x)):
        for i in range(j):
            matrix[i, j] = matrix[j, i] = stats.kendalltau(x[i], x[j]).statistic
    return np.stack([matrix ** 2 if kwargs.get("squared") else matrix.copy() for _, kwargs in specs])


def fresh_data(raw):
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(raw.copy(), normalise=False, detrend=False)


def benchmark(baseline, raw, specs, repeats):
    methods = {"baseline": lambda d: run(baseline, specs, d),
               "refactor": lambda d: run(basic, specs, d)}
    if specs[0][0] == "KendallTau":
        methods["unique_pair_scipy"] = lambda d: unique_scipy_kendall(specs, d)
    times = {name: [] for name in methods}
    outputs = {}
    for method in methods.values():
        method(fresh_data(raw[:3, :128]))
    for repeat in range(repeats):
        # Alternate which implementation runs first to reduce ordering bias.
        names = list(methods) if repeat % 2 == 0 else list(methods)[::-1]
        for name in names:
            data = fresh_data(raw)
            gc.collect()
            start = time.perf_counter()
            result = methods[name](data)
            times[name].append(time.perf_counter() - start)
            outputs[name] = result
        for name in methods:
            np.testing.assert_allclose(outputs[name], outputs["baseline"], rtol=5e-13, atol=5e-14)
    medians = {name: float(np.median(values)) for name, values in times.items()}
    finite = np.isfinite(outputs["baseline"]) & np.isfinite(outputs["refactor"])
    return {"shape": list(raw.shape), "specifications": specs, "seconds": times,
            "median_seconds": medians,
            "speedup": medians["baseline"] / medians["refactor"],
            "maximum_absolute_difference": float(np.max(abs(outputs["baseline"][finite] - outputs["refactor"][finite]), initial=0)),
            "nan_masks_equal": bool(np.array_equal(np.isnan(outputs["baseline"]), np.isnan(outputs["refactor"])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="c28512b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "tests/basic_refactor_results.json")
    args = parser.parse_args()
    baseline, revision = load_baseline(args.baseline)
    workloads = {
        "kendall": [("KendallTau", {})],
        "kendall_both": [("KendallTau", {"squared": squared}) for squared in (False, True)],
        "spearman_both": [("SpearmanR", {"squared": squared}) for squared in (False, True)],
        "cross_correlation_eight": [("CrossCorrelation", {"statistic": statistic, "squared": squared, "sigonly": sigonly})
                                   for squared in (False, True) for statistic in ("max", "mean") for sigonly in (False, True)],
    }
    raw = np.random.default_rng(324).normal(size=(118, 1024))
    raw[1] += .6 * raw[0]
    raw = stats.zscore(raw, axis=1, ddof=1)
    results = {}
    with threadpool_limits(limits=1):
        for name, specs in workloads.items():
            results[name] = benchmark(baseline, raw, specs, args.repeats)
            print(name, results[name]["median_seconds"], "speedup", results[name]["speedup"], flush=True)
        pools = threadpool_info()
    sources = ["pyspi/statistics/basic.py", "pyspi/_rank.py", "pyspi/_cache.py", "pyspi/_spectral.py"]
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "baseline_commit": revision, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
              "protocol": {"repeats": args.repeats, "seed": 324, "thread_limit": 1,
                           "timed": "Matrix calculations, including cache validation and population; excludes preprocessing and startup.",
                           "comparison": "Fresh Data each repeat; ordinary/squared/summary variants share Data within a workload.",
                           "unique_pair_scipy": "Unique-pair scipy.stats.kendalltau calls, one matrix shared by both variants; isolates the coefficient-only kernel's additional gain.",
                           "limits": "Synthetic data on this computer; not cohort throughput."},
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
              "threadpools": pools, "measurements": results}
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
