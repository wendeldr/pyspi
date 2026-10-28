"""Time the complete amplitude candidate set and its shared increments.

Reproduce with .venv/bin/python tests/benchmark_amplitude_selection.py.
Selection comes from the exhaustive review of reviewed.yaml, not a hand list.
"""

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import time
import warnings

from review_amplitude_selection import (
    ROOT, classify, instantiate, make_data, np, resolve_metrics, synthetic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone-only", action="store_true",
                        help="Append independent costs without repeating completed family timings.")
    args = parser.parse_args()
    metrics = [metric for metric in resolve_metrics(ROOT / "pyspi/workflow/reviewed.yaml", sampling_rate_hz=2048)
               if classify(metric) in ("amplitude_distance", "amplitude_extremum", "shrinkage_precision_structure")]
    assert len(metrics) == 12
    source = synthetic(channels=118, samples=1024, seed=9027) * 1e-5
    source -= source.mean(axis=0, keepdims=True)
    path = ROOT / "tests/amplitude_full_selection_timings.json"
    if args.standalone_only:
        report = json.loads(path.read_text())
        assert report["identifiers"] == [metric.identifier for metric in metrics]
        passes = {metric.identifier: [] for metric in metrics}
        for repetition in range(5):
            for metric in (metrics if repetition % 2 == 0 else metrics[::-1]):
                data = make_data(source, False)
                spi = instantiate(metric)
                start = time.perf_counter()
                result = spi.multivariate(data)
                duration = time.perf_counter()-start
                assert np.isfinite(result[~np.eye(118, dtype=bool)]).all()
                passes[metric.identifier].append(duration)
        report["standalone"] = {
            "design": "Same unnormalized synthetic CAR workload; five alternating-order passes, a fresh Data/cache for EVERY output; no other fit or metric supplies cached work. Preprocessing/imports/constructors/I/O excluded.",
            "passes_seconds": passes,
            "summary": {identifier: {"median_seconds": float(np.median(seconds)),
                                     "seconds_600": float(np.median(seconds))*600,
                                     "seconds_41400": float(np.median(seconds))*41400}
                        for identifier, seconds in passes.items()},
        }
        path.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report["standalone"]["summary"], indent=2))
        return
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "design": "118x1024 synthetic unequal-amplitude AR/noise/sinusoid/spike mixture; seed 9027; CAR, linear detrend; one numerical thread; fresh Data per timed family. Five alternating normalized/unnormalized repetitions after one untimed warmup. Preprocessing, imports and I/O excluded. Serial projections, not measured patient/cohort throughput.",
        "identifiers": [metric.identifier for metric in metrics],
        "passes": {"normalized": [], "unnormalized": []},
    }
    for normalized in (True, False):
        data = make_data(source, normalized)
        for metric in metrics:
            instantiate(metric).multivariate(data)
    for repetition in range(5):
        for normalized in ((True, False) if repetition % 2 == 0 else (False, True)):
            data = make_data(source, normalized)
            stages, finite, captured = {}, {}, []
            for metric in metrics:
                spi = instantiate(metric)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    start = time.perf_counter()
                    result = spi.multivariate(data)
                    stages[metric.identifier] = time.perf_counter()-start
                finite[metric.identifier] = int(np.isfinite(result[~np.eye(118, dtype=bool)]).sum())
                captured.extend(str(warning.message) for warning in caught)
            report["passes"]["normalized" if normalized else "unnormalized"].append({
                "seconds": stages, "finite_offdiagonal_entries": finite, "warnings": dict(Counter(captured))})
    report["summary"] = {}
    for label, passes in report["passes"].items():
        total = float(np.median([sum(row["seconds"].values()) for row in passes]))
        stages = {metric.identifier: float(np.median([row["seconds"][metric.identifier] for row in passes])) for metric in metrics}
        group = {mechanism: float(np.median([sum(row["seconds"][metric.identifier] for metric in metrics if classify(metric) == mechanism) for row in passes]))
                 for mechanism in ("amplitude_distance", "amplitude_extremum", "shrinkage_precision_structure")}
        report["summary"][label] = {"seconds_per_window": total, "seconds_600": total*600,
                                    "seconds_41400": total*41400, "median_stage_seconds": stages,
                                    "median_group_seconds": group}
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
