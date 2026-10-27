"""Fresh-window xcorr timing against the last committed implementation.

Run: .venv/bin/python -B tests/benchmark_xcorr.py
The definitions intentionally differ: this measures end-to-end cost, not
numerical equivalence. Independent direct-sum references live in test_xcorr.py.
"""

import datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from benchmark_basic import load_baseline, fresh_data

import numpy as np
import scipy
from scipy import stats
from threadpoolctl import threadpool_limits
from pyspi.statistics.basic import CrossCorrelation, CrossCorrelationUnavailableWarning


def main():
    old, revision = load_baseline("3d993e0")
    raw = np.random.default_rng(324).normal(size=(118, 1024))
    raw[1] += .6 * raw[0]
    raw = stats.zscore(raw, axis=1, ddof=1)
    specs = [(stat, sq, sig) for sig in (False, True) for stat in ("max", "mean") for sq in (False, True)]
    workloads = {
        "four_all_lags": [s for s in specs if not s[2]],
        "four_significant_lags": [s for s in specs if s[2]],
        "eight_original_variants": specs,
        "twelve_with_min_and_peak": specs + [(s, False, sig) for s in ("min", "peak") for sig in (False, True)],
    }
    results = {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", CrossCorrelationUnavailableWarning)
        for name, specifications in workloads.items():
            implementations = {"refactor": (CrossCorrelation, {"fs": 2048})}
            if not name.startswith("twelve"):
                implementations["baseline"] = (old.CrossCorrelation, {})
            instances = {name: [cls(statistic=stat, squared=sq, sigonly=sig, **extra)
                                for stat, sq, sig in specifications]
                         for name, (cls, extra) in implementations.items()}
            times = {name: [] for name in instances}
            for metrics in instances.values():
                d = fresh_data(raw[:3])
                for metric in metrics:
                    metric.multivariate(d)
            for repeat in range(3):
                order = list(instances) if repeat % 2 else list(instances)[::-1]
                for method in order:
                    d = fresh_data(raw)
                    gc.collect()
                    start = time.perf_counter()
                    for metric in instances[method]:
                        metric.multivariate(d)
                    times[method].append(time.perf_counter() - start)
            medians = {key: float(np.median(value)) for key, value in times.items()}
            row = {"specifications": specifications, "seconds": times, "median_seconds": medians,
                   "refactor_extrapolated_seconds": {str(n): n * medians["refactor"] for n in (1, 600, 600 * 69)}}
            if "baseline" in medians:
                row["speedup"] = medians["baseline"] / medians["refactor"]
            results[name] = row
            print(name, row, flush=True)
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_commit": revision, "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
        "shape": [118, 1024], "sampling_rate_hz": 2048, "maximum_lag_seconds": .125,
        "protocol": "Three alternating repeats, fresh Data each run; variants share caches; one numerical thread; excludes imports, preprocessing, I/O. Synthetic data. Changed numerical and significance definitions; timings compare workloads, not identical results.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("pyspi/_xcorr.py", "pyspi/statistics/basic.py", "tests/benchmark_xcorr.py")},
        "measurements": results,
    }
    (ROOT / "tests/xcorr_refactor_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
