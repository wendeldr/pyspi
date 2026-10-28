"""Review every configured output under z-scoring and changes of amplitude units.

This is a targeted selection audit, not a clinical validation or a golden update.
Run with .venv/bin/python tests/review_amplitude_selection.py --config PATH --out PATH.
"""

import argparse
from collections import Counter
import contextlib
import datetime
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import detrend, lfilter
from pyspi.data import Data
from pyspi.workflow._metrics import resolve_metrics


def make_data(values, normalise=False):
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(values.copy(), normalise=normalise, detrend=True)


def instantiate(metric):
    return getattr(importlib.import_module(metric.module, "pyspi"), metric.class_name)(**metric.parameters)


def classify(metric):
    """Mechanisms from source/formulas; probes challenge these classifications."""
    cls, params = metric.class_name, metric.parameters
    if cls in ("Covariance", "Precision"):
        estimator = params.get("estimator", "EmpiricalCovariance")
        if estimator in ("EmpiricalCovariance", "MinCovDet", "EllipticEnvelope"):
            return "covariance_rescaling" if cls == "Covariance" else "inverse_or_pseudoinverse_scaling"
        if estimator in ("LedoitWolf", "OAS", "ShrunkCovariance"):
            return "shrinkage_covariance_weighting" if cls == "Covariance" else "shrinkage_precision_structure"
        if estimator == "GraphicalLasso":
            return "fixed_graphical_penalty"
        if estimator == "GraphicalLassoCV":
            return "cross_validated_graphical_structure"
        raise ValueError(estimator)
    if cls == "PairwiseDistance":
        name = params["metric"]
        return "scale_invariant" if name == "cosine" else "signed_braycurtis" if name == "braycurtis" else "amplitude_distance"
    if cls == "Barycenter":
        return "amplitude_extremum"
    if cls == "LinearModel":
        return "bayesian_prior_units" if params["model"] == "BayesianRidge" else "fixed_regression_penalty"
    known_invariant = {
        "SpearmanR", "KendallTau", "CrossCorrelation", "RegressionErrorCausalInference",
        "MutualInfo", "TimeLaggedMutualInfo", "TransferEntropy", "PowerEnvelopeCorrelation",
        "CoherencePhase", "CoherenceMagnitude", "ImaginaryCoherence", "PhaseSlopeIndex",
        "PhaseLockingValue", "PhaseLagIndex", "SignedPhaseLagIndex", "WeightedPhaseLagIndex",
        "SignedWeightedPhaseLagIndex", "DebiasedSquaredPhaseLagIndex",
        "DebiasedSquaredWeightedPhaseLagIndex", "PairwisePhaseConsistency",
    }
    if cls not in known_invariant:
        raise ValueError(f"Unreviewed class: {cls}")
    return "scale_invariant"


def degree(metric):
    cls, params = metric.class_name, metric.parameters
    if cls in ("Covariance", "Precision"):
        return (2 if cls == "Covariance" else -2) * (2 if params.get("squared", False) else 1)
    if cls == "LinearModel":
        return 2
    if cls == "Barycenter":
        return 2 if params.get("squared", False) else 1
    if cls == "PairwiseDistance" and params["metric"] in ("euclidean", "cityblock", "chebyshev"):
        return 1
    return 0


def compare(actual, expected):
    mask = ~np.eye(len(actual), dtype=bool)
    a, b = actual[mask], expected[mask]
    finite = np.isfinite(a) & np.isfinite(b)
    result = {"finite_pairs": int(finite.sum()),
              "nan_mask_equal": bool(np.array_equal(np.isnan(a), np.isnan(b))),
              "infinity_mask_equal": bool(np.array_equal(np.isinf(a), np.isinf(b)))}
    if finite.any():
        result["max_abs_error"] = float(np.max(abs(a[finite] - b[finite])))
        denominator = float(np.linalg.norm(b[finite]))
        result["relative_frobenius_error"] = (
            float(np.linalg.norm(a[finite] - b[finite]) / denominator) if denominator else None)
    return result


def synthetic(channels=8, samples=1024, seed=9025):
    rng = np.random.default_rng(seed)
    x = lfilter([1], [1, -.65], rng.normal(size=(channels, samples)), axis=1)
    x[1] += .65*x[0]
    x[2, 3:] += .6*x[0, :-3]
    x[3] -= .7*x[1]
    time_axis = np.arange(samples)/2048
    for index in range(channels):
        x[index] += .3*np.sin(2*np.pi*(6 + 7*index)*time_axis + index/3)
    x[0, 400:409] += np.array([0, 1, 3, 5, 8, 5, 3, 1, 0])
    x[4, 400:409] += .7*x[0, 400:409]
    x = detrend(x, axis=1)
    x /= x.std(axis=1, ddof=1)[:, None]
    return x * np.geomspace(.25, 12, channels)[:, None]


def run_all(metrics, data):
    matrices, details = {}, {}
    for metric in metrics:
        # Control FAST-MCD's search for the scaling comparison only; do not
        # confuse stochastic support changes with amplitude information.
        np.random.seed(9026)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            try:
                spi = instantiate(metric)
                value = spi.multivariate(data)
                matrices[metric.identifier] = value
                entries = value[~np.eye(len(value), dtype=bool)]
                detail = {"finite": int(np.isfinite(entries).sum()), "nan": int(np.isnan(entries).sum()),
                          "inf": int(np.isinf(entries).sum()), "zero": int((entries == 0).sum())}
                if metric.class_name in ("Covariance", "Precision"):
                    fit = spi._from_cache(data)
                    for attr in ("shrinkage_", "alpha_", "n_iter_"):
                        if hasattr(fit, attr):
                            detail[attr] = float(getattr(fit, attr))
            except Exception as exc:
                detail = {"error": f"{type(exc).__name__}: {exc}"}
        detail["seconds"] = time.perf_counter() - start
        detail["warnings"] = dict(Counter(f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught))
        details[metric.identifier] = detail
    return matrices, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "pyspi/workflow/reviewed.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "tests/amplitude_full_review_results.json")
    args = parser.parse_args()
    path = args.config
    metrics = resolve_metrics(path, sampling_rate_hz=2048)
    assert metrics and len(metrics) == len({m.identifier for m in metrics})
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inventory_count": len(metrics),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design": "8x1024 synthetic AR/noise/sinusoid/spike mixture; signal seed 9025; estimator seed 9026. Full rank and common-average-referenced data. Detrending retained. Compare z-score, unequal channel amplitudes, volts (x1e-5), microvolts (x10). One numerical thread. This is formula/invariance validation, not EZ validation or patient timing.",
        "mechanism_counts": dict(Counter(classify(metric) for metric in metrics)),
        "metrics": {m.identifier: {"class": m.class_name, "module": m.module, "parameters": m.parameters,
                                    "mechanism": classify(m), "source": {}} for m in metrics},
    }
    original = synthetic()
    for reference in ("full_rank", "car"):
        values = original.copy()
        if reference == "car":
            values -= values.mean(axis=0, keepdims=True)
        raw_data = make_data(values)
        sds = raw_data.to_numpy(squeeze=True).std(axis=1, ddof=1)
        matrices, details = {}, {}
        for label, gain, normalize in (("normalized", 1., True), ("unequal", 1., False),
                                       ("volts", 1e-5, False), ("microvolts", 10., False)):
            print(reference, label, flush=True)
            matrices[label], details[label] = run_all(metrics, make_data(values*gain, normalize))
        for metric in metrics:
            identifier = metric.identifier
            row = {"runs": {label: detail[identifier] for label, detail in details.items()}}
            for label in ("unequal", "volts", "microvolts"):
                if identifier in matrices[label] and identifier in matrices["normalized"]:
                    row[f"{label}_versus_normalized"] = compare(matrices[label][identifier], matrices["normalized"][identifier])
            if identifier in matrices["volts"] and identifier in matrices["microvolts"]:
                row["unit_change_after_removing_output_units"] = compare(
                    matrices["microvolts"][identifier] / (1e6**degree(metric)), matrices["volts"][identifier])
            if identifier in matrices["unequal"] and identifier in matrices["normalized"]:
                value = matrices["unequal"][identifier]
                if metric.class_name in ("Covariance", "Precision"):
                    factors = sds[:, None]*sds[None, :]
                    exponent = 2 if metric.parameters.get("squared", False) else 1
                    if metric.class_name == "Precision":
                        exponent *= -1
                    row["after_removing_pair_scale"] = compare(value/factors**exponent, matrices["normalized"][identifier])
                elif metric.class_name == "LinearModel":
                    row["after_removing_target_variance"] = compare(value/sds[None, :]**2, matrices["normalized"][identifier])
                    for label, gain in (("volts", 1e-5), ("microvolts", 10.)):
                        if identifier in matrices[label]:
                            variance = np.broadcast_to(sds[None, :]**2 * gain**2 * 1023/1024, value.shape)
                            row[f"{label}_versus_target_variance"] = compare(matrices[label][identifier], variance)
            report["metrics"][identifier]["source"][reference] = row
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failures = []
    for identifier, entry in report["metrics"].items():
        if entry["mechanism"] != "scale_invariant":
            continue
        for reference, row in entry["source"].items():
            for label in ("unequal", "volts", "microvolts"):
                check = row.get(f"{label}_versus_normalized", {})
                if (not check.get("nan_mask_equal", False) or not check.get("infinity_mask_equal", False)
                        or check.get("max_abs_error", 0.) > 1e-9):
                    failures.append([identifier, reference, label, check])
    report["invariance_failures"] = failures
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"counts": report["mechanism_counts"], "invariance_failures": failures}, indent=2))


if __name__ == "__main__":
    main()
