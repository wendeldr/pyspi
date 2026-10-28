"""Runtime gate for reviewed additions versus originally retained related metrics.

This benchmarks spectral, PEC and cross-correlation selections, not all 185
historical metrics. Unchanged metric families and I/O are excluded explicitly.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from pyspi.data import Data
from pyspi.statistics import basic, misc, spectral
from pyspi._spectral import directed_workspace


def selections(revision):
    text = subprocess.check_output(["git", "show", f"{revision}:pyspi/workflow/retained185.yaml"], cwd=ROOT, text=True)
    old = yaml.safe_load(text)
    baseline = []
    for name, entry in old[".statistics.spectral"].items():
        if name != "GroupDelay":
            baseline += [getattr(spectral, name)(**kwargs) for kwargs in entry["configs"]]
    baseline += [misc.PowerEnvelopeCorrelation(**kwargs) for kwargs in old[".statistics.misc"]["PowerEnvelopeCorrelation"]["configs"]]
    # The EEG workflow resolves xcorr's fs to its actual target sampling rate.
    baseline += [basic.CrossCorrelation(fs=2048, **kwargs) for kwargs in old[".statistics.basic"]["CrossCorrelation"]["configs"]]
    cheap = []
    for name, original in (("SignedPhaseLagIndex", "PhaseLagIndex"),
                           ("SignedWeightedPhaseLagIndex", "WeightedPhaseLagIndex")):
        cheap += [getattr(spectral, name)(**kwargs) for kwargs in old[".statistics.spectral"][original]["configs"]]
    cheap += [basic.CrossCorrelation(statistic=statistic, sigonly=False, fs=2048) for statistic in ("min", "peak")]
    cheap += [basic.CrossCorrelation(statistic=statistic, squared=squared, sigonly=True, fs=2048)
              for statistic, squared in (("max", False), ("max", True), ("mean", False),
                                         ("mean", True), ("min", False), ("peak", False))]
    bands = old[".statistics.spectral"]["PhaseLagIndex"]["configs"]
    classes = ("DirectedCoherence", "DirectedTransferFunction", "PartialDirectedCoherence",
               "GeneralizedPartialDirectedCoherence", "DirectDirectedTransferFunction",
               "SpectralGrangerCausality")
    all_directed = [getattr(spectral, name)(**kwargs) for name in classes for kwargs in bands]
    ddtf = [spi for spi in all_directed if isinstance(spi, spectral.DirectDirectedTransferFunction)]
    return baseline, {"signed_phase_and_xcorr": cheap, "ddtf": ddtf, "directed_family": all_directed}


def run(raw, baseline, additions):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        data = Data(raw.copy())
        for spi in (*baseline, *additions):
            register = getattr(spi, "_register_shared_work", None)
            if register is not None:
                register(data)
        stages, finite = {}, {}
        for name, spis in (("baseline", baseline), ("addition", additions)):
            start = time.perf_counter()
            for spi in spis:
                result = spi.multivariate(data)
                finite[spi.identifier] = int(np.isfinite(result).sum())
            stages[name] = time.perf_counter()-start
        diagnostics = None
        if any(isinstance(spi, spectral.DirectDirectedTransferFunction) for spi in additions):
            conn = directed_workspace(data, 1)
            reason, residual, iterations = conn._fit_diagnostics
            i, j = np.triu_indices(raw.shape[0], 1)
            codes, counts = np.unique(reason[0, i, j], return_counts=True)
            diagnostics = dict(reason_code_counts={str(int(c)): int(k) for c, k in zip(codes, counts)},
                               maximum_reconstruction_error=float(np.nanmax(residual)),
                               iteration_quantiles=np.quantile(iterations[0, i, j], [0, .5, .9, 1]).tolist(),
                               retained_summary_cache_bytes=conn._arrays.nbytes)
        return dict(seconds=stages, finite_entries=finite, fit_diagnostics=diagnostics,
                    warning_counts={kind: sum(type(w.message).__name__ == kind for w in emitted)
                                    for kind in sorted({type(w.message).__name__ for w in emitted})})


def interleaved_check(raw, baseline, additions, repeats):
    """Time the final config's adjacent signed/unsigned grouping separately.

    The initial append-only experiment can evict earlier phase intermediates.
    This check reflects inserting signed variants beside their unsigned family.
    It also times fresh baseline-only passes to expose measurement variability.
    """
    ordered, remaining = [], list(additions)
    signed_for = {spectral.PhaseLagIndex: spectral.SignedPhaseLagIndex,
                  spectral.WeightedPhaseLagIndex: spectral.SignedWeightedPhaseLagIndex}
    for position, spi in enumerate(baseline):
        ordered.append(("baseline", spi))
        next_type = type(baseline[position+1]) if position+1 < len(baseline) else None
        if type(spi) in signed_for and next_type is not type(spi):
            selected = [item for item in remaining if type(item) is signed_for[type(spi)]]
            ordered.extend(("signed_phase", item) for item in selected)
            remaining = [item for item in remaining if item not in selected]
    ordered.extend(("xcorr", item) for item in remaining)

    def timed(sequence):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = Data(raw.copy())
            elapsed = {"baseline": 0., "signed_phase": 0., "xcorr": 0.}
            for label, spi in sequence:
                start = time.perf_counter()
                spi.multivariate(data)
                elapsed[label] += time.perf_counter()-start
        return elapsed

    results = []
    for repetition in range(repeats):
        times = {}
        for name in (("baseline", "combined") if repetition % 2 == 0 else ("combined", "baseline")):
            sequence = [("baseline", spi) for spi in baseline] if name == "baseline" else ordered
            times[name] = timed(sequence)
            print(f"Interleaved check {repetition+1} {name}: {times[name]}", flush=True)
        results.append(times)
    stages = {key: float(np.median([row["combined"][key] for row in results]))
              for key in ("baseline", "signed_phase", "xcorr")}
    baseline_only = float(np.median([row["baseline"]["baseline"] for row in results]))
    total = float(np.median([sum(row["combined"].values()) for row in results]))
    added = stages["signed_phase"]+stages["xcorr"]
    return dict(raw_seconds=results, median_combined_stage_seconds=stages,
                baseline_only_median_seconds=baseline_only, combined_median_seconds=total,
                difference_of_medians_seconds=total-baseline_only,
                timed_added_calls_sequential_seconds={"one_window": added, "600_windows": added*600,
                                                      "600_windows_x_69": added*600*69},
                protocol="Final config grouping: signed phase families immediately after their unsigned families; cross-correlation additions share existing curves. Fresh baseline-only and combined passes in alternating order. Per-stage times and difference of total medians are distinct observations; noisy timing limits precise subtraction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-revision", default="cb60808")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT/"tests/review_selection_results.json")
    parser.add_argument("--interleaved-only", action="store_true",
                        help="Append final cheap-variant ordering check; do not repeat directed benchmarks.")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.baseline_revision], cwd=ROOT, text=True).strip()
    baseline, additions = selections(revision)
    raw = np.random.default_rng(732).normal(size=(118, 1024))
    raw[1, 1:] += .7*raw[0, :-1]
    if args.interleaved_only:
        report = json.loads(args.output.read_text())
        report["interleaved_selection"] = interleaved_check(raw, baseline, additions["signed_phase_and_xcorr"], args.repeats)
        report["interleaved_selection"]["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        return
    results = {name: [] for name in additions}
    report = dict(shape=list(raw.shape), seed=732, original_selection_revision=revision,
                  baseline_identifiers=[spi.identifier for spi in baseline],
                  additions={name: [spi.identifier for spi in spis] for name, spis in additions.items()},
                  runs=results, protocol="Fresh Data per workload; one numerical-library thread; three alternating-order repeats. Baseline consists of originally retained related spectral/PEC/xcorr metrics evaluated with corrected current code, with GD excluded as agreed. Not the entire 185-metric run. Preprocessing and I/O excluded; synthetic data, representative 118-channel/1024-sample window.",
                  environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "spectral-connectivity")},
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                      ("pyspi/_spectral.py", "pyspi/_wilson.py", "pyspi/_directed.py", "pyspi/_envelope.py",
                       "pyspi/statistics/spectral.py", "pyspi/statistics/misc.py", "pyspi/statistics/basic.py",
                       "tests/benchmark_review_selection.py")})
    names = list(additions)
    for repetition in range(args.repeats):
        for name in (names if repetition % 2 == 0 else names[::-1]):
            print(f"Starting {name}, repeat {repetition+1}", flush=True)
            result = run(raw, baseline, additions[name])
            results[name].append(result)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(f"Finished {name}: {result['seconds']}", flush=True)
    report["medians"] = {}
    for name, runs in results.items():
        baseline_time = float(np.median([run["seconds"]["baseline"] for run in runs]))
        addition_time = float(np.median([run["seconds"]["addition"] for run in runs]))
        report["medians"][name] = dict(baseline_seconds=baseline_time, added_seconds=addition_time,
            combined_seconds=baseline_time+addition_time, added_fraction_of_baseline=addition_time/baseline_time,
            added_sequential_seconds={"one_window": addition_time, "600_windows": 600*addition_time,
                                      "600_windows_x_69": 600*69*addition_time})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["medians"], indent=2), flush=True)


if __name__ == "__main__":
    main()
