"""Compare production regressions with the committed pairwise implementation.

Run: .venv/bin/python tests/benchmark_regression.py
Inputs and thread limits match between methods. Historical code is loaded only
inside this benchmark; it is not a production compatibility path.
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
import tracemalloc
import types

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
startup = time.perf_counter()
import numpy as np
import scipy
import sklearn
from scipy.stats import zscore
from threadpoolctl import threadpool_info, threadpool_limits
from pyspi.data import Data
from pyspi.statistics import causal, misc

IMPORT_SECONDS = time.perf_counter() - startup


def baseline(revision):
    modules = {}
    for name in ("causal", "misc"):
        source = subprocess.check_output(
            ["git", "show", f"{revision}:pyspi/statistics/{name}.py"], cwd=ROOT, text=True)
        module = types.ModuleType(f"regression_baseline_{name}")
        exec(compile(source, f"<baseline {name}>", "exec"), module.__dict__)
        modules[name] = module
    return modules


def new_data(raw):
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(raw.copy(), normalise=False, detrend=False)


def run(modules, names, data):
    result, seconds = {}, {}
    for name in names:
        spi = (modules["causal"].RegressionErrorCausalInference() if name == "RECI"
               else modules["misc"].LinearModel(name))
        start = time.perf_counter()
        result[name] = spi.multivariate(data)
        seconds[name] = time.perf_counter()-start
    return result, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="31993ac")
    parser.add_argument("--models", nargs="+", default=["Ridge", "RECI"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "tests/regression_refactor_results.json")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.baseline], cwd=ROOT, text=True).strip()
    methods = {"baseline": baseline(revision), "refactor": {"causal": causal, "misc": misc}}
    rng = np.random.default_rng(1001)
    raw = rng.normal(size=(118, 1024))
    raw[1] += .7*raw[0]
    raw = zscore(raw, axis=1, ddof=1)
    results = {}
    load_start = os.getloadavg()
    with threadpool_limits(limits=1):
        for modules in methods.values():
            run(modules, args.models, new_data(raw[:3, :100]))
        for names in [[name] for name in args.models] + [args.models]:
            label = "+".join(names)
            times = {name: [] for name in methods}
            outputs = {}
            for repeat in range(args.repeats):
                order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
                for method in order:
                    data = new_data(raw)
                    gc.collect()
                    outputs[method], elapsed = run(methods[method], names, data)
                    times[method].append(elapsed)
                    print(label, repeat, method, elapsed, flush=True)
            differences = {}
            for name in names:
                a, b = outputs["baseline"][name], outputs["refactor"][name]
                np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)
                differences[name] = float(np.nanmax(abs(a-b)))
            data = new_data(raw)
            tracemalloc.start()
            run(methods["refactor"], names, data)
            retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cache = data._calculation_cache
            median = {method: float(np.median([sum(row.values()) for row in rows]))
                      for method, rows in times.items()}
            results[label] = {
                "seconds": times, "total_median_seconds": median,
                "speedup": median["baseline"]/median["refactor"],
                "incremental_median_seconds": {
                    method: {name: float(np.median([row[name] for row in rows])) for name in names}
                    for method, rows in times.items()},
                "max_absolute_difference": differences,
                "refactor_memory": {"tracemalloc_retained_bytes": retained, "tracemalloc_peak_bytes": peak,
                                    "snapshot_bytes": cache.source.nbytes,
                                    "derived_cache_bytes": cache.families["moments"].arrays.nbytes},
                "projected_seconds": {method: {str(n): seconds*n for n in (1, 600, 600*69)}
                                      for method, seconds in median.items()},
            }
        pools = threadpool_info()
    sources = ["pyspi/_moments.py", "pyspi/_regression.py", "pyspi/statistics/misc.py", "pyspi/statistics/causal.py"]
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_commit": revision, "shape": list(raw.shape), "seed": 1001,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "threadpools": pools,
        "import_seconds": IMPORT_SECONDS, "load_start": load_start, "load_end": os.getloadavg(),
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        "protocol": {"repeats": args.repeats, "timed": "Fresh matrix calls including cache population; preprocessing and startup excluded.",
                     "sharing": "Each workload shares one fresh Data across its metrics; per-stage timings give incremental cost.",
                     "memory": "Separate untimed tracemalloc pass; not whole-process RSS or peak native-library allocations.",
                     "limits": "Synthetic window on this computer. Serial extrapolations exclude I/O and do not promise cohort throughput."},
        "measurements": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
