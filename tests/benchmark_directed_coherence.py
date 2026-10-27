"""Time the DC correction against a committed implementation on fresh Data.

The committed implementation is only a benchmark comparator, never a fallback.
Formula correctness is established independently in test_directed_coherence.py.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from benchmark_phase_sync import ROOT, baseline, peak, run
import numpy as np
from pyspi.statistics import spectral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="923b9f6")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/directed_coherence_results.json")
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    old, revision = baseline(args.baseline)
    raw = np.random.default_rng(8026).normal(size=(4, 1024))
    raw[1] = .8*np.roll(raw[0], 3) + .2*raw[1]
    summaries = [dict(fmin=lo, fmax=hi, statistic=statistic)
                 for lo, hi in [(0, .5), (0, .25), (.25, .5)] for statistic in ("mean", "max")]
    workloads = {
        "dc_full_band": [("DirectedCoherence", {})],
        "dc_six_summaries": [("DirectedCoherence", kwargs) for kwargs in summaries],
        "five_directed_estimators": [(name, {}) for name in
             ("DirectedCoherence", "DirectedTransferFunction", "PartialDirectedCoherence",
              "GeneralizedPartialDirectedCoherence", "DirectDirectedTransferFunction")],
    }
    report = dict(
        baseline_revision=revision, seed=8026, shape=list(raw.shape), repeats=args.repeats,
        numerical_library_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
        protocol="Fresh Data; warmup; alternating baseline/candidate order; seven-run medians by default. "
                 "Includes transform, ordered-pair fits, estimator and band summaries. Allocation peaks "
                 "are separate tracemalloc runs, not RSS. Intentional formula change: standard squared "
                 "source-variance-weighted DC; stable amplitude scaling before squaring. Other directed "
                 "formulas and factorization are unchanged. Timings are not a 118-channel projection.",
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                       ("pyspi/_spectral.py", "pyspi/statistics/spectral.py", "tests/benchmark_directed_coherence.py")},
        workloads={},
    )
    for name, specs in workloads.items():
        modules = {"baseline": old, "candidate": spectral}
        for module in modules.values():
            run(module, raw, specs)
        walls = {key: [] for key in modules}
        for repeat in range(args.repeats):
            for key in (list(modules) if repeat % 2 == 0 else list(reversed(modules))):
                gc.collect()
                start = time.perf_counter()
                run(modules[key], raw, specs)
                walls[key].append(time.perf_counter()-start)
        median = {key: float(np.median(values)) for key, values in walls.items()}
        result = dict(median_s=median, walls_s=walls,
                      candidate_to_baseline_ratio=median['candidate']/median['baseline'],
                      peak_traced_mib={key: peak(module, raw, specs) for key, module in modules.items()})
        report['workloads'][name] = result
        print(name, json.dumps(result), flush=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == "__main__":
    main()
