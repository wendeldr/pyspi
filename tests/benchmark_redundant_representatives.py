"""Choose equivalent normalized outputs using independent, uncached costs.

No fit/cache from another selected calculation is used. Fast alternatives have
11 alternating-order passes; robust alternatives have three matched-seed passes.
"""

from collections import Counter
import datetime
import json
import time
import warnings

from review_amplitude_selection import ROOT, compare, make_data, np, synthetic
from pyspi.statistics.basic import Covariance, Precision
from pyspi.statistics.distance import PairwiseDistance


def evaluate(metric, values, *, seed=9028):
    data = make_data(values, True)
    np.random.seed(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        result = metric.multivariate(data)
        duration = time.perf_counter()-start
    return result, duration, dict(Counter(f"{w.category.__name__}: {str(w.message)[:180]}" for w in caught))


def main():
    values = synthetic(channels=118, samples=1024, seed=9028)
    values -= values.mean(axis=0, keepdims=True)
    fast = [PairwiseDistance(metric="cosine"), Covariance(),
            Covariance(estimator="ShrunkCovariance"), Covariance(squared=True),
            Covariance(estimator="ShrunkCovariance", squared=True)]
    robust = [Precision(estimator="MinCovDet"), Precision(estimator="EllipticEnvelope")]
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "design": "118x1024 synthetic seed 9028; CAR followed by detrend and ddof=1 z-score; one numerical thread. Every timed output uses a fresh Data/cache and no other metric. Preprocessing, imports, constructors and I/O excluded. Timing order alternates. Robust fits use the same seed for like-for-like algorithm comparison.",
              "passes": {metric.identifier: [] for metric in fast+robust}, "checks": {}}
    # Warm imports/numerical kernels, but never reuse a Data or fitted estimator.
    for metric in fast:
        evaluate(metric, values)
    last = {}
    for repeat in range(11):
        for metric in (fast if repeat % 2 == 0 else fast[::-1]):
            value, duration, captured = evaluate(metric, values)
            last[metric.identifier] = value
            report["passes"][metric.identifier].append({"seconds": duration, "warnings": captured})
    n = values.shape[1]
    report["checks"]["cosine_equals_one_minus_scaled_empirical"] = compare(
        last[fast[0].identifier], 1-last[fast[1].identifier]*n/(n-1))
    report["checks"]["shrunk_equals_point9_empirical"] = compare(last[fast[2].identifier], .9*last[fast[1].identifier])
    report["checks"]["shrunk_squared_equals_point81_empirical_squared"] = compare(last[fast[4].identifier], .81*last[fast[3].identifier])
    for repeat in range(3):
        for metric in (robust if repeat % 2 == 0 else robust[::-1]):
            print("Robust pass", repeat+1, metric.identifier, flush=True)
            value, duration, captured = evaluate(metric, values, seed=9028+repeat)
            last[metric.identifier] = value
            report["passes"][metric.identifier].append({"seconds": duration, "warnings": captured})
            print("Seconds", duration, flush=True)
        report["checks"][f"matched_seed_robust_pass_{repeat+1}"] = compare(last[robust[0].identifier], last[robust[1].identifier])
    report["standalone_summary"] = {}
    for identifier, passes in report["passes"].items():
        seconds = [row["seconds"] for row in passes]
        median = float(np.median(seconds))
        report["standalone_summary"][identifier] = {
            "median_seconds": median, "min_seconds": min(seconds), "max_seconds": max(seconds),
            "seconds_600": median*600, "seconds_41400": median*41400}
    (ROOT / "tests/redundant_representative_timings.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("checks", "standalone_summary")}, indent=2))


if __name__ == "__main__":
    main()
