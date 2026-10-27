"""Measure shared phase-estimator work against a committed implementation.

Run: .venv/bin/python -B tests/benchmark_phase_sync.py --repeats 3
The baseline is isolated benchmark code, never a production fallback. Ordinary
PLI/wPLI definitions intentionally change; independent scalar/pair-product tests
in test_phase_lag.py establish correctness, not the old implementation.
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


def baseline(revision):
    revision = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    modules = []
    for filename in ("pyspi/_spectral.py", "pyspi/statistics/spectral.py"):
        source = subprocess.check_output(["git", "show", f"{revision}:{filename}"], cwd=ROOT, text=True)
        module = types.ModuleType("phase_benchmark_baseline_" + str(len(modules)))
        exec(compile(source, f"<{revision}:{filename}>", "exec"), module.__dict__)
        modules.append(module)
    for name in ("spectral_workspace", "data_spectral_cache", "frequency_mask", "validate_frequencies"):
        setattr(modules[1], name, getattr(modules[0], name))
    return modules[1], revision


def run(module, raw, specs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = Data(raw.copy(), normalise=False)
        return [getattr(module, name)(**kwargs).multivariate(data) for name, kwargs in specs]


def peak(module, raw, specs):
    gc.collect()
    tracemalloc.start()
    run(module, raw, specs)
    _, memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return memory / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="6e36cc6")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zero-policy", action="store_true",
                        help="Measure zero-sign/denominator fixes, including standalone and shared dsPLI.")
    parser.add_argument("--coherence-phase", action="store_true",
                        help="Compare arithmetic and circular coherence-phase summaries.")
    parser.add_argument("--cpu", type=int, help="Keep this benchmark process on one allowed CPU.")
    parser.add_argument("--variants-only", action="store_true",
                        help="Add a paired current-code six/eight estimator comparison to the existing report.")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/phase_sync_refactor_results.json")
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    old, revision = baseline(args.baseline)
    raw = np.random.default_rng(8026).normal(size=(118, 1024))
    raw[1] = .8 * np.roll(raw[0], 3) + .2 * raw[1]
    classes = ["PhaseLockingValue", "PhaseLagIndex", "WeightedPhaseLagIndex",
               "DebiasedSquaredPhaseLagIndex", "DebiasedSquaredWeightedPhaseLagIndex",
               "PairwisePhaseConsistency"]
    family = [(name, dict(fmin=low, fmax=high, statistic=statistic))
              for name in classes for low, high in ((0, .5), (0, .25), (.25, .5))
              for statistic in ("mean", "max")]
    # Match catalogue order: signed variants immediately follow their unsigned versions.
    with_signed = []
    for name in classes:
        with_signed += [spec for spec in family if spec[0] == name]
        if name in ("PhaseLagIndex", "WeightedPhaseLagIndex"):
            with_signed += [("Signed"+cls, kwargs) for cls, kwargs in family if cls == name]
    workloads = {
        "pli": [("PhaseLagIndex", {})],
        "wpli": [("WeightedPhaseLagIndex", {})],
        "debiased_wpli": [("DebiasedSquaredWeightedPhaseLagIndex", {})],
        "six_estimators_36_summaries": family,
        "eight_estimators_48_summaries": with_signed,
    }
    if args.zero_policy:
        workloads = {
            "pli": [("PhaseLagIndex", {})],
            "debiased_pli": [("DebiasedSquaredPhaseLagIndex", {})],
            "pli_and_debiased_pli": [("PhaseLagIndex", {}), ("DebiasedSquaredPhaseLagIndex", {})],
            "wpli": [("WeightedPhaseLagIndex", {})],
            "debiased_wpli": [("DebiasedSquaredWeightedPhaseLagIndex", {})],
            "six_estimators_36_summaries": family,
        }
    if args.coherence_phase:
        workloads = {
            "coherence_phase_full_band": [("CoherencePhase", {})],
            "coherence_phase_three_bands": [("CoherencePhase", dict(fmin=low, fmax=high))
                                             for low, high in ((0, .5), (0, .25), (.25, .5))],
        }
    if args.variants_only:
        report = json.loads(args.output.read_text())
        run(spectral, raw[:3, :128], with_signed)
        walls = {"six_estimators": [], "with_signed_variants": []}
        specs = {"six_estimators": family, "with_signed_variants": with_signed}
        for repeat in range(args.repeats):
            order = list(specs) if repeat % 2 == 0 else list(reversed(specs))
            for name in order:
                gc.collect()
                start = time.perf_counter()
                run(spectral, raw, specs[name])
                walls[name].append(time.perf_counter()-start)
        differences = np.subtract(walls["with_signed_variants"], walls["six_estimators"])
        result = dict(walls_s=walls, paired_extra_seconds=differences.tolist(),
                      median_extra_seconds=float(np.median(differences)),
                      median_extra_fraction=float(np.median(differences / walls["six_estimators"])),
                      median_s={name: float(np.median(times)) for name, times in walls.items()},
                      protocol="Current code only; fresh Data; alternate six/eight workload order; "
                               "report per-repeat differences so thermal/timing drift is visible.",
                      source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                     for name in report["source_sha256"]})
        report["paired_signed_variants"] = result
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return
    report = dict(baseline_revision=revision, seed=8026, shape=list(raw.shape),
                  python=platform.python_version(), numpy=np.__version__, repeats=args.repeats,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  numerical_library_threads=1,
                  protocol="Fresh Data per workload; warmup on a small input; alternating baseline/candidate "
                           "order; medians of repeated calls including Data construction. Peak memory is a "
                           "separate tracemalloc run, not RSS. PLI/wPLI magnitude definitions intentionally "
                           "differ. The eight-estimator workload compares new with/without signed variants, "
                           "using the original six-estimator baseline for context.",
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ("pyspi/_cache.py", "pyspi/_spectral.py",
                                              "pyspi/statistics/spectral.py")}, workloads={})
    if args.zero_policy:
        report["protocol"] = (
            "Fresh Data per workload; small warmup; alternating baseline/candidate order; "
            "medians including Data construction; separate traced-memory runs, not RSS. "
            "Intentional definition corrections: zero signs contribute zero to dsPLI "
            "pair products; weighted zero denominators return NaN; amplitude cutoff removed. "
            "Scaled prefix sums stabilize debiased wPLI, and upper-triangle phase tiles "
            "reuse reversed channel pairs. "
            "Compare with --baseline 8b66a4a to isolate these changes.")
    if args.coherence_phase:
        report["protocol"] = (
            "Fresh Data per workload; small warmup; alternating baseline/candidate order; "
            "medians including Data construction; separate traced-memory runs, not RSS. "
            "Intentional definition change: equal-weight circular mean replaces arithmetic "
            "angle mean. Zero coherency is omitted; undefined resultant returns NaN. "
            "Compare with --baseline 923b9f6 to isolate this change.")
    for name, candidate_specs in workloads.items():
        old_specs = family if name == "eight_estimators_48_summaries" else candidate_specs
        cases = {"baseline": (old, old_specs), "candidate": (spectral, candidate_specs)}
        for module, specs in cases.values():
            run(module, raw[:3, :128], specs)
        walls = {key: [] for key in cases}
        for repeat in range(args.repeats):
            order = list(cases) if repeat % 2 == 0 else list(reversed(cases))
            for key in order:
                gc.collect()
                start = time.perf_counter()
                run(cases[key][0], raw, cases[key][1])
                walls[key].append(time.perf_counter()-start)
        medians = {key: float(np.median(values)) for key, values in walls.items()}
        result = dict(walls_s=walls, median_s=medians,
                      speedup=medians["baseline"]/medians["candidate"],
                      peak_traced_mib={key: peak(module, raw, specs)
                                       for key, (module, specs) in cases.items()})
        report["workloads"][name] = result
        print(name, json.dumps(result), flush=True)
    if not (args.zero_policy or args.coherence_phase):
        a = report["workloads"]["six_estimators_36_summaries"]["median_s"]["candidate"]
        b = report["workloads"]["eight_estimators_48_summaries"]["median_s"]["candidate"]
        report["unpaired_signed_variants_observed_seconds"] = b-a
        report["unpaired_signed_variants_observed_fraction"] = b/a-1
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
