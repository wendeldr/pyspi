"""Descriptive novelty and surrogate checks; no prediction or significance claims.

Uses uncommitted cached EEG windows from benchmark.py. Surrogates are generated
on original filtered sources before rereferencing, or shifted after referencing
to distinguish induced shared-reference structure. All band views use the same
per-channel shifts/phases as the base window. No labels/outcomes are read.
"""

import argparse
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from benchmark import apply_reference, data_for
from pyspi.workflow import load_config
from pyspi.workflow._metrics import resolve_metrics
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow._reference import reference_plan


def instantiate(specs):
    return [getattr(importlib.import_module(s.module), s.class_name)(**s.parameters) for s in specs]


def mean_windows(values):
    values = np.asarray(values)
    n = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)


def rank_similarity(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 4 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return None
    return float(spearmanr(a[mask], b[mask]).statistic)


def describe(values):
    selected = values[~np.eye(len(values), dtype=bool)]
    finite = selected[np.isfinite(selected)]
    return {"finite_fraction": float(len(finite)/len(selected)),
            "mean": float(finite.mean()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "q90": float(np.quantile(finite, .9)) if len(finite) else None}


def shifted(values, shifts):
    indices = (np.arange(values.shape[1])[None, :]-shifts[:, None]) % values.shape[1]
    return np.take_along_axis(values, indices, axis=1)


def surrogate(arrays, rng, kind):
    p, n = arrays[0].shape
    if kind == "shift":
        shifts = rng.integers(n//4, 3*n//4, size=p)
        return [shifted(values, shifts) for values in arrays]
    phase = np.exp(1j*rng.uniform(-np.pi, np.pi, size=(p, n//2+1)))
    phase[:, 0] = 1
    if n % 2 == 0:
        phase[:, -1] = 1
    return [np.fft.irfft(np.fft.rfft(values, axis=1)*phase, n=n, axis=1) for values in arrays]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("at least one surrogate replicate is required")
    specs = resolve_metrics(ROOT / "pyspi/workflow/additional_candidate_pilot.yaml", sampling_rate_hz=2048)
    candidates = instantiate(specs)
    current = resolve_metrics(ROOT / "pyspi/workflow/reviewed.yaml", sampling_rate_hz=2048)
    baseline_specs = [s for s in current if (
        s.class_name in ("SpearmanR", "KendallTau", "RegressionErrorCausalInference", "MutualInfo", "PowerEnvelopeCorrelation")
        or s.class_name == "TransferEntropy" and s.parameters.get("estimator") == "gaussian"
        or s.class_name == "Covariance" and s.parameters.get("estimator") == "EmpiricalCovariance"
        or s.class_name == "PairwiseDistance" and s.parameters.get("metric") == "cosine"
        or s.class_name == "CrossCorrelation" and s.parameters.get("statistic") == "peak" and not s.parameters.get("sigonly", True)
        or s.class_name == "PhaseLockingValue" and s.parameters.get("fmin") == 30/2048 and s.parameters.get("fmax") == 70/2048)]
    baselines = instantiate(baseline_specs)
    result = {"seed": 20261028, "replicates": args.replicates, "patients": {},
        "baseline_ids": [m.identifier for m in baselines],
        "controls": {"shift_before_reference": "Independent circular source shifts, preserving each input's marginal samples/autocorrelation before window preprocessing and preserving within-channel band relationships.",
                     "phase_before_reference": "Independent Fourier phase randomization per source, shared across its views. Preserves periodogram, but changes marginal shape and within-channel nonlinear structure.",
                     "shift_after_reference": "Independent circular output shifts, removing synchronous shared-reference structure while preserving each referenced input's samples/autocorrelation."},
        "limits": "Descriptive three-patient/three-window pilot, not p-values or predictive evidence. Circular shifts may retain stationary narrowband PAC magnitude. Histograms measure marginal shape, so these coupling nulls do not test histogram usefulness. Detrending can change the post-surrogate sample distribution. Baseline comparison is a selected existing panel, not all 188 metrics."}

    def save():
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    with numerical_threads(1):
        for pid in ("004", "001", "111"):
            loaded = np.load(args.cache / f"{pid}-inputs.npz")
            names = loaded["channel_names"].tolist()
            bands = tuple(map(tuple, loaded["bands"]))
            raw = [(base, {band: loaded[f"band_{i}"][w] for i, band in enumerate(bands)})
                   for w, base in enumerate(loaded["base"])]
            local_cfg = load_config(ROOT / f"tests/candidate_pilot/configs/{pid}-laplacian.yaml")
            local_plan = reference_plan(names, local_cfg.reference)
            available = np.flatnonzero(local_plan[1])
            # Spread contacts over the recording and retain an adjacent triple.
            probes = np.unique(np.r_[available[:3], available[np.rint(np.linspace(0, len(available)-1, 13)).astype(int)]])
            entry = {"probe_contacts": [names[i] for i in probes], "references": {}}
            result["patients"][pid] = entry
            matrices = {}
            for method in ("car", "laplacian"):
                cfg = load_config(ROOT / f"tests/candidate_pilot/configs/{pid}-{method}.yaml")
                plan = reference_plan(names, cfg.reference)
                valid = np.flatnonzero(plan[1])
                observed = np.load(args.cache / f"{pid}-{method}-results.npz")
                matrices[method] = {m.identifier: mean_windows(observed[m.identifier]) for m in candidates}
                records = {"baseline_similarity": {}, "probe_subset_max_difference": {}, "surrogates": {}, "observed": {}}
                entry["references"][method] = records
                referenced = [(apply_reference(base, names, cfg.reference, plan),
                    {band: apply_reference(values, names, cfg.reference, plan) for band, values in views.items()}) for base, views in raw]
                baseline_values = {m.identifier: [] for m in baselines}
                for base, views in referenced:
                    data, usable = data_for(base, views, valid, cfg, ())
                    for metric in baselines:
                        with redirect_stdout(io.StringIO()):
                            values = metric.multivariate(data)
                        full = np.full((len(names), len(names)), np.nan)
                        full[np.ix_(usable, usable)] = values
                        baseline_values[metric.identifier].append(full)
                averaged = {key: mean_windows(values) for key, values in baseline_values.items()}
                for metric in candidates:
                    comparisons = [(key, rank_similarity(matrices[method][metric.identifier], values)) for key, values in averaged.items()]
                    comparisons = sorted(((key, rho) for key, rho in comparisons if rho is not None), key=lambda x: -abs(x[1]))
                    records["baseline_similarity"][metric.identifier] = [{"baseline": key, "spearman_rho": rho} for key, rho in comparisons[:3]]
                    records["surrogates"][metric.identifier] = {key: [] for key in result["controls"]}
                    records["observed"][metric.identifier] = []
                for window, ((base, views), (source, source_views)) in enumerate(zip(referenced, raw)):
                    data, usable = data_for(base, views, probes, cfg, bands)
                    for metric in candidates:
                        with redirect_stdout(io.StringIO()):
                            values = metric.multivariate(data)
                        expected = observed[metric.identifier][window][np.ix_(usable, usable)]
                        np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-11)
                        records["probe_subset_max_difference"][metric.identifier] = max(
                            records["probe_subset_max_difference"].get(metric.identifier, 0.), float(np.nanmax(abs(values-expected))))
                        records["observed"][metric.identifier].append(describe(values))
                    for control in result["controls"]:
                        for replicate in range(args.replicates):
                            # Common random numbers isolate reference/order effects.
                            rng = np.random.default_rng(np.random.SeedSequence(
                                [result["seed"], int(pid), window, replicate]))
                            arrays = ([base]+[views[band] for band in bands] if control == "shift_after_reference"
                                      else [source]+[source_views[band] for band in bands])
                            transformed = surrogate(arrays, rng, "phase" if control.startswith("phase") else "shift")
                            if control != "shift_after_reference":
                                transformed = [apply_reference(a, names, cfg.reference, plan) for a in transformed]
                            data, _ = data_for(transformed[0], dict(zip(bands, transformed[1:])), probes, cfg, bands)
                            for metric in candidates:
                                with redirect_stdout(io.StringIO()):
                                    values = metric.multivariate(data)
                                records["surrogates"][metric.identifier][control].append(describe(values))
                    save()
                    print(pid, method, "window", window, "surrogates complete", flush=True)
            entry["matched_reference_similarity"] = {m.identifier: rank_similarity(matrices["car"][m.identifier], matrices["laplacian"][m.identifier]) for m in candidates}
            # Shaft-average PAC is exploratory only; compare same contact probes.
            shaft_ref = {**local_cfg.reference, "method": "shaft_average"}
            plan = reference_plan(names, shaft_ref)
            pac = next(m for m in candidates if m.identifier.startswith("pac_"))
            values = []
            for source, views in raw:
                base = apply_reference(source, names, shaft_ref, plan)
                referenced_views = {b: apply_reference(a, names, shaft_ref, plan) for b, a in views.items()}
                data, usable = data_for(base, referenced_views, probes, local_cfg, bands)
                values.append(pac.multivariate(data))
            entry["shaft_average_pac"] = {"probe_window_summaries": [describe(v) for v in values],
                "rho_with_car_on_same_contacts": rank_similarity(mean_windows(values), matrices["car"][pac.identifier][np.ix_(usable, usable)]),
                "rho_with_laplacian_on_same_contacts": rank_similarity(mean_windows(values), matrices["laplacian"][pac.identifier][np.ix_(usable, usable)])}
            save()


if __name__ == "__main__":
    main()
