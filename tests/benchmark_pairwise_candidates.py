"""Bounded formula-stage comparison; workflow/recording pilots are separate.

Example: .venv/bin/python tests/benchmark_pairwise_candidates.py
  --input /tmp/pyspi-kernel-eeg-window.npy --out /tmp/candidate-math.json
Use one numerical thread. Input is a previously prepared channels-by-time EEG
window, not raw EDF data. Never launch the full quadratic baseline at 118 nodes.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from hyppo.independence import Dcorr
from tslearn.metrics import dtw
from pyspi.data import Data
from pyspi._moments import moment_workspace
from pyspi.statistics.distance import DistanceCorrelation, DynamicTimeWarping, HistogramRelativeEntropy
from pyspi.statistics.misc import InterDependenceScore


def data(x):
    with redirect_stdout(io.StringIO()):
        return Data(x.copy(), dim_order="ps", normalise=False)


def measure(fn, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        value = fn()
        times.append(perf_counter() - start)
    median = float(np.median(times))
    return value, {"samples_seconds": times, "median_seconds": median,
                   "seconds_600": median * 600, "seconds_43200": median * 43200}


def old(x, method):
    # Historical Undirected dispatch evaluated both orders before mirroring.
    result = np.full((len(x), len(x)), np.nan)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                result[i, j] = (Dcorr(bias=False).statistic(x[i, :, None], x[j, :, None])
                                if method == "dcorr" else dtw(x[i], x[j], global_constraint="sakoe_chiba"))
    i, j = np.tril_indices(len(x), -1)
    result[i, j] = result[j, i]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    x = np.load(args.input)
    if x.ndim != 2:
        raise ValueError("input must be channels by observations")
    result = {"input_path": str(args.input), "shape": list(x.shape), "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                  ("pyspi/_distance_correlation.py", "pyspi/_histogram.py", "pyspi/_dtw.py", "pyspi/statistics/misc.py")},
              "load_average_start": list(os.getloadavg()), "standalone": {}, "small_comparisons": {},
              "limits": "Single prepared EEG window. Standalone includes fresh Data and own caches. Imports, compilation, continuous preprocessing and output excluded. Baseline one pass at eight channels; native medians of three passes. Serial projections, not measured cohort throughput."}
    def save():
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    dtw_spi = DynamicTimeWarping(global_constraint="sakoe_chiba")
    start = perf_counter()
    dtw_spi.multivariate(data(x[:2, :31]))
    old(x[:2, :31], "dcorr")
    old(x[:2, :31], "dtw")
    result["jit_and_reference_warmup_seconds"] = perf_counter() - start
    for name, spi in [("dcorr", DistanceCorrelation()), ("dtw", dtw_spi)]:
        expected, baseline = measure(lambda: old(x[:8], name), 1)
        actual, native = measure(lambda: spi.multivariate(data(x[:8])), 3)
        mask = np.isfinite(expected)
        result["small_comparisons"][name] = {"baseline": baseline, "native": native,
            "max_absolute_difference": float(np.max(abs(actual[mask] - expected[mask]))),
            "matching_masks": bool(np.array_equal(np.isfinite(actual), np.isfinite(expected)))}
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        save()
    for spi in [HistogramRelativeEntropy(), DistanceCorrelation(), DistanceCorrelation(biased=True), dtw_spi, InterDependenceScore()]:
        _, timing = measure(lambda: spi.multivariate(data(x)), 3)
        result["standalone"][spi.identifier] = timing
        save()
        print(spi.identifier, timing["median_seconds"], flush=True)
    d = data(x)
    start = perf_counter()
    DistanceCorrelation().multivariate(d)
    cold = perf_counter() - start
    start = perf_counter()
    DistanceCorrelation(biased=True).multivariate(d)
    extra = perf_counter() - start
    result["distance_family"] = {"cold_first_seconds": cold, "actual_increment_biased_seconds": extra,
                                 "complete_family_seconds": cold + extra,
                                 "derived_cache_bytes": moment_workspace(d).arrays.nbytes,
                                 "distance_block_bytes": moment_workspace(d).block_bytes}
    result["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["load_average_end"] = list(os.getloadavg())
    save()


if __name__ == "__main__":
    main()
