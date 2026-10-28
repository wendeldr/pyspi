"""Reproduce isolated group-delay significance diagnostics; no production edits.

The original timings and calibration were run as inline scripts. This file
preserves their algorithms for reproduction, without repeating those runs.
Benchmark p-values are synthetic, not inferred from the saved EEG outputs.
"""

import argparse
import json
import time

import numpy as np
from scipy.stats import false_discovery_control, norm
from spectral_connectivity.statistics import (
    Benjamini_Hochberg_procedure,
    coherence_bias,
    fisher_z_transform,
    get_normal_distribution_p_values,
)


def per_pair(p, alpha=.05):
    """Prototype for finite p-values shaped (windows, frequencies, pairs)."""
    x = np.ascontiguousarray(np.moveaxis(p, 1, -1))
    ordered = np.sort(x, axis=-1)
    rank = alpha * np.arange(1, x.shape[-1] + 1) / x.shape[-1]
    cutoff = np.max(np.where(ordered <= rank, ordered, -1), axis=-1, keepdims=True)
    return np.moveaxis(x <= cutoff, -1, 1)


def benchmark():
    rng = np.random.default_rng(2706)
    results = []
    for nf in (35, 125, 513):
        p = rng.uniform(size=(1, nf, 6903))
        p[..., ::8] *= 1e-4
        np.testing.assert_array_equal(
            per_pair(p[:, :, :8]), false_discovery_control(p[:, :, :8], axis=1) <= .05)
        times = {"pooled": [], "pairwise": []}
        funcs = {"pooled": Benjamini_Hochberg_procedure, "pairwise": per_pair}
        for fn in funcs.values():
            fn(p)
        for repeat in range(5):
            names = ("pooled", "pairwise") if repeat % 2 == 0 else ("pairwise", "pooled")
            for name in names:
                start = time.perf_counter()
                funcs[name](p)
                times[name].append(time.perf_counter() - start)
        results.append(dict(frequencies=nf, pairs=6903, seconds=times,
                            medians={k: float(np.median(v)) for k, v in times.items()}))
    return results


def calibration():
    rng = np.random.default_rng(271027)
    n, k = 200000, 5
    z = rng.normal(size=(n, k, 2)) + 1j * rng.normal(size=(n, k, 2))
    c = np.sum(z[:, :, 0] * z[:, :, 1].conj(), axis=1) / np.sqrt(
        np.sum(abs(z[:, :, 0])**2, axis=1) * np.sum(abs(z[:, :, 1])**2, axis=1))
    old = get_normal_distribution_p_values(fisher_z_transform(c, coherence_bias(k)))
    proposed = (1 - abs(c)**2)**(k - 1)
    threshold = np.tanh(coherence_bias(k) + np.sqrt(coherence_bias(k)) * norm.isf(.05))
    return dict(seed=271027, replicates=n, independent_complex_observations=k,
                nominal_alpha=.05, old_rejection_fraction=float(np.mean(old <= .05)),
                beta_null_rejection_fraction=float(np.mean(proposed <= .05)),
                old_magnitude_threshold=float(threshold),
                old_exact_null_rejection_probability=float((1 - threshold**2)**(k - 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("benchmark", "calibration"), required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.part](), indent=2))
