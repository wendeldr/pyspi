"""Compare the spectral backend with a specified committed wrapper implementation.

Run: .venv/bin/python -B tests/benchmark_spectral.py
The baseline is loaded from git into an isolated module, never installed as a
production fallback. Both implementations use the same installed dependencies.
PSI, group-delay spacing, and shared frequency selection intentionally differ
from that baseline. Mathematical checks live in the spectral unit tests.
"""

import argparse
import contextlib
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
import warnings

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from pyspi.data import Data
from pyspi.statistics import spectral


def load_baseline(revision):
    revision = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:pyspi/statistics/spectral.py"], cwd=ROOT, text=True)
    module = types.ModuleType("spectral_benchmark_baseline")
    exec(compile(source, "<committed spectral baseline>", "exec"), module.__dict__)
    return module, revision


def run(module, raw, specs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = Data(raw.copy(), normalise=False)
        return np.stack([getattr(module, name)(**kwargs).multivariate(data) for name, kwargs in specs])


def peak(module, raw, specs):
    gc.collect()
    tracemalloc.start()
    run(module, raw, specs)
    _, maximum = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return maximum / 2**20


def benchmark(baseline, raw, specs, repeats):
    # Warm both code paths with a small input, outside measurements.
    for module in (baseline, spectral):
        run(module, raw[:3, :128], specs)
    walls = {"baseline": [], "candidate": []}
    for _ in range(repeats):
        for name, module in (("baseline", baseline), ("candidate", spectral)):
            gc.collect()
            start = time.perf_counter()
            values = run(module, raw, specs)
            walls[name].append(time.perf_counter() - start)
            if name == "baseline":
                expected = values
            else:
                actual = values
    checks = []
    for index, (name, kwargs) in enumerate(specs):
        reason = "inclusive bands and complete one-sided frequency grid"
        if name == "PhaseSlopeIndex":
            reason = "adjacent-bin PSI instead of thinned all-frequency pairs; inclusive band"
        elif name == "GroupDelay":
            reason = "transform-derived regression spacing; inclusive band"
        elif name == "DirectDirectedTransferFunction":
            reason += "; Nyquist included in full-frequency normalization"
        elif name == "SpectralGrangerCausality":
            reason += "; requested DC is no longer clamped away"
        checks.append({
            "measure": name, "parameters": kwargs,
            "comparison": "intentional change: " + reason,
            "correctness_tests": ["tests/unit/test_spectral.py", "tests/unit/test_spectral_bands.py",
                                  "tests/unit/test_psi.py"],
        })
    finite = np.isfinite(expected) & np.isfinite(actual)
    return {
        "shape": list(raw.shape), "measures": specs, "walls_s": walls,
        "baseline_median_s": float(np.median(walls["baseline"])),
        "candidate_median_s": float(np.median(walls["candidate"])),
        "speedup_median": float(np.median(walls["baseline"]) / np.median(walls["candidate"])),
        "max_abs_difference": float(np.max(abs(actual[finite]-expected[finite]))) if finite.any() else None,
        "nan_masks_equal": bool(np.array_equal(np.isnan(actual), np.isnan(expected))),
        "numerical_checks": checks,
        "finite_values_per_measure": [int(np.isfinite(value).sum()) for value in actual],
        "peak_traced_mib_baseline": peak(baseline, raw, specs),
        "peak_traced_mib_candidate": peak(spectral, raw, specs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="d088a58ead76c9df757ce972497c295a30a8266b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/spectral_band_refactor_results.json")
    parser.add_argument("--workload", help="Run only the named workload.")
    args = parser.parse_args()
    baseline, revision = load_baseline(args.baseline)
    raw = np.random.default_rng(8026).normal(size=(118, 1024))
    raw[1] = .8 * np.roll(raw[0], 3) + .2 * raw[1]
    bands = [(0, .5), (0, .25), (.25, .5)]
    mixed = [("CoherenceMagnitude", {}), ("CoherencePhase", {})]
    mixed += [(name, dict(fmin=lo, fmax=hi, **extra))
              for name, extra in [("PhaseSlopeIndex", {}), ("GroupDelay", {"statistic": "delay"})]
              for lo, hi in bands]
    directed = [(name, {}) for name in ("DirectedCoherence", "PartialDirectedCoherence",
                "GeneralizedPartialDirectedCoherence", "DirectedTransferFunction",
                "DirectDirectedTransferFunction", "SpectralGrangerCausality")]
    workloads = {
        "coherence_118x1024": (raw, [("CoherenceMagnitude", {})]),
        "shared_eight_118x1024": (raw, mixed),
        "weighted_phase_lag_118x1024": (raw, [("WeightedPhaseLagIndex", {})]),
        "directed_six_4x1024": (raw[:4], directed),
        "group_delay_three_bands_118x1024": (raw, [("GroupDelay", dict(fmin=lo, fmax=hi, statistic="delay")) for lo, hi in bands]),
        "psi_three_bands_118x1024": (raw, [("PhaseSlopeIndex", dict(fmin=lo, fmax=hi)) for lo, hi in bands]),
    }
    if args.workload:
        if args.workload not in workloads:
            parser.error(f"unknown workload {args.workload!r}; choose from {list(workloads)}")
        workloads = {args.workload: workloads[args.workload]}
    report = {
        "baseline_revision": revision, "python": platform.python_version(), "numpy": np.__version__,
        "repeats": args.repeats, "seed": 8026, "numerical_library_threads": 1,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ("pyspi/_spectral.py", "pyspi/statistics/spectral.py")},
        "protocol": "Paired baseline/candidate calls on fresh Data; small warmup; median of repeats. "
                    "Includes Data construction and all selected measures. Peak memory is a separate "
                    "tracemalloc run (NumPy allocations, not process RSS). Intentional differences: "
                    "adjacent-bin PSI, transform-derived group-delay spacing, inclusive bands "
                    "and Nyquist throughout, and Granger includes requested DC. Mathematical "
                    "correctness is checked by the spectral unit tests; these timings compare "
                    "different frequency-selection policies and are not numerical equivalence claims.",
        "workloads": {},
    }
    for name, (values, specs) in workloads.items():
        report["workloads"][name] = result = benchmark(baseline, values, specs, args.repeats)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps({key: result[key] for key in (
            "speedup_median", "max_abs_difference", "peak_traced_mib_baseline", "peak_traced_mib_candidate")}), flush=True)


if __name__ == "__main__":
    main()
