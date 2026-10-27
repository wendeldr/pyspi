"""Directed-spectral formula checks and read-only factorization experiments.

Run: .venv/bin/python -B audit/checks/probes/spectral_directed_followup.py
Exact VAR spectra separate formula errors from estimation/factorization errors.
Padding and stricter convergence are experiments, not approved defaults.
"""

import contextlib
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from spectral_connectivity.minimum_phase_decomposition import minimum_phase_decomposition

from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi.data import Data
from pyspi.statistics import spectral


def adjoint(a):
    return a.swapaxes(-1, -2).conj()


def exact_model(a, covariance, nfft=1024):
    ar = np.eye(len(a)) - np.exp(-2j*np.pi*np.fft.fftfreq(nfft))[:, None, None]*a
    transfer = np.linalg.inv(ar)
    return transfer, transfer @ covariance @ adjoint(transfer)


def connection_from_model(transfer, covariance):
    nfft, p, _ = transfer.shape
    conn = SpectralConnectivity(np.zeros((1, 1, 5, nfft, p), complex),
                                np.fft.fftfreq(nfft), np.array([0.]))
    conn.__dict__["_transfer_function"] = transfer[None, :nfft//2+1]
    conn.__dict__["_noise_covariance"] = covariance[None]
    return conn


def formula_checks():
    a = np.array([[.6, 0], [.9, .3]])
    covariance = np.diag([4., .25])
    transfer, spectrum = exact_model(a, covariance)
    conn = connection_from_model(transfer, covariance)
    h = transfer[:513]
    power = abs(h)**2
    weighted = power * np.diag(covariance)[None, None, :]
    dc = weighted / weighted.sum(axis=-1, keepdims=True)
    ar_power = abs(np.linalg.inv(h))**2
    gp_power = ar_power / np.diag(covariance)[None, :, None]
    refs = {
        "directed_coherence": dc,
        "directed_transfer_function": power / power.sum(axis=-1, keepdims=True),
        "partial_directed_coherence": ar_power / ar_power.sum(axis=-2, keepdims=True),
        "generalized_partial_directed_coherence": gp_power / gp_power.sum(axis=-2, keepdims=True),
    }
    errors = {name: float(np.max(abs(getattr(conn, name)()[0]-ref)))
              for name, ref in refs.items()}
    for name, error in errors.items():
        assert error < 1e-12, (name, error)
    actual = conn.directed_coherence()[0]
    # The installed dependency is retained only as an audit comparator.
    dependency_dc = sc.Connectivity.directed_coherence(conn)[0]
    assert dependency_dc[:, 1, 0].max() > 1
    assert np.all((actual >= 0) & (actual <= 1))
    # Recover an analytic spectrum before interpreting finite-window problems.
    factor = minimum_phase_decomposition(spectrum[None])
    h0 = np.fft.ifft(factor, axis=1).real[:, 0]
    fitted_h = factor @ np.linalg.inv(h0)[:, None]
    exact_factor_error = float(np.max(abs(fitted_h[0]-transfer)))
    assert exact_factor_error < 1e-6
    return dict(
        ar_matrix=a.tolist(), innovation_covariance=covariance.tolist(),
        max_abs_formula_errors=errors,
        dependency_dc_max_offdiagonal=float(dependency_dc[:, 1, 0].max()),
        current_dc_max_offdiagonal=float(actual[:, 1, 0].max()),
        squared_dc_max_offdiagonal=float(dc[:, 1, 0].max()),
        squared_dc_max_row_sum_error=float(np.max(abs(dc.sum(axis=-1)-1))),
        exact_spectrum_transfer_recovery_error=exact_factor_error,
    )


def raw_fixture():
    raw = np.zeros((3, 8192))
    noise = np.random.default_rng(821).normal(size=raw.shape)
    for t in range(1, raw.shape[1]):
        raw[0, t] = .5*raw[0, t-1] + noise[0, t]
        raw[1, t] = .2*raw[1, t-1] + .7*raw[0, t-1] + noise[1, t]
        raw[2, t] = noise[2, t]
    return raw[:, -2048:]


def order_checks(raw):
    data = Data(raw, normalise=False, detrend=False)
    workspace = spectral_workspace(data, 1)
    forward, reverse = workspace.pair(0, 1), workspace.pair(1, 0)
    names = ["directed_coherence", "directed_transfer_function", "partial_directed_coherence",
             "generalized_partial_directed_coherence", "direct_directed_transfer_function"]
    pair_gaps = {name: float(np.max(abs(forward.measure(name)-reverse.measure(name)[..., ::-1, ::-1])))
                 for name in names}
    order = [2, 1, 0]
    permuted_data = Data(raw[order], normalise=False, detrend=False)
    classes = [spectral.DirectedCoherence, spectral.DirectedTransferFunction,
               spectral.PartialDirectedCoherence, spectral.GeneralizedPartialDirectedCoherence,
               spectral.DirectDirectedTransferFunction, spectral.SpectralGrangerCausality]
    wrapper_gaps = {}
    for cls in classes:
        spi = cls(fmin=.01, fmax=.2)
        before = spi.multivariate(data)[np.ix_(order, order)]
        after = spi.multivariate(permuted_data)
        wrapper_gaps[cls.__name__] = dict(max_abs_difference=float(np.nanmax(abs(before-after))),
                                         nan_mask_equal=bool(np.array_equal(np.isnan(before), np.isnan(after))))
    return dict(ordered_pair_per_bin_gaps=pair_gaps, full_wrapper_permutation=wrapper_gaps)


def padding_check(raw, multiplier, tolerance, iterations):
    n = raw.shape[1]
    transform = sc.Multitaper(raw.T, n_fft_samples=n*multiplier)
    spectrum = SpectralConnectivity._mean_cross_spectrum(transform.fft())
    factors = [minimum_phase_decomposition(s, tolerance=tolerance, max_iterations=iterations)
               for s in (spectrum, spectrum[..., ::-1, ::-1])]
    residuals, dtfs = [], []
    for factor, spec in zip(factors, (spectrum, spectrum[..., ::-1, ::-1])):
        residuals.append(float(np.max(abs(factor@adjoint(factor)-spec))/np.max(abs(spec))))
        zero = np.fft.ifft(factor, axis=1).real[:, 0]
        power = abs((factor @ np.linalg.inv(zero)[:, None])[:, :n*multiplier//2+1])**2
        dtfs.append(power / power.sum(axis=-1, keepdims=True))
    gap = abs(dtfs[0]-dtfs[1][..., ::-1, ::-1])
    return dict(nfft=n*multiplier, tolerance=tolerance, max_iterations=iterations,
                reconstruction_relative_errors=residuals, max_dtf_order_gap=float(gap.max()),
                dtf_order_gap_on_original_grid=float(gap[:, ::multiplier].max()))


def conditioning_check():
    # A1 -> B1 -> C1, no direct A1 -> C1 coefficient.
    a = np.array([[.5, 0, 0], [.7, .2, 0], [0, .6, .3]])
    covariance = np.eye(3)
    transfer, spectrum = exact_model(a, covariance)
    conn = connection_from_model(transfer, covariance)
    full = conn.partial_directed_coherence()[0, :, 2, 0]
    pair_spectrum = spectrum[:, [0, 2]][:, :, [0, 2]]
    factor = minimum_phase_decomposition(pair_spectrum[None])
    zero = np.fft.ifft(factor, axis=1).real[:, 0]
    marginal_transfer = (factor @ np.linalg.inv(zero)[:, None])[0]
    pair = connection_from_model(marginal_transfer, (zero@adjoint(zero))[0])
    marginal = pair.partial_directed_coherence()[0, :, 1, 0]
    assert abs(full).max() < 1e-20
    return dict(full_three_channel_pdc_a_to_c_max=float(full.max()),
                marginal_two_channel_pdc_a_to_c_mean=float(marginal.mean()),
                observations_per_frequency=5, full_118_channel_spectrum_rank_bound=5)


def timing_checks(raw, repeats):
    """One two-channel transform and shared factorization feeding five metrics."""
    names = ["directed_coherence", "directed_transfer_function", "partial_directed_coherence",
             "generalized_partial_directed_coherence", "direct_directed_transfer_function"]

    def run(multiplier):
        transform = sc.Multitaper(raw.T, n_fft_samples=raw.shape[1]*multiplier)
        conn = SpectralConnectivity(transform.fft(), transform.frequencies, transform.time)
        for name in names:
            conn.measure(name)

    times = {str(m): [] for m in (1, 2, 4)}
    for m in (1, 2, 4):
        run(m)
    for repeat in range(repeats):
        for m in ((1, 2, 4) if repeat % 2 == 0 else (4, 2, 1)):
            start = time.perf_counter()
            run(m)
            times[str(m)].append(time.perf_counter()-start)
    medians = {m: float(np.median(values)) for m, values in times.items()}
    return dict(shape=list(raw.shape), repeats=repeats, raw_seconds=times,
                median_seconds=medians, ratio_to_unpadded={m: t/medians['1'] for m, t in medians.items()},
                protocol="Fresh two-channel transform, one factorization, five metrics; small warmup; "
                         "alternating padding order; one numerical-library thread. These are experiment "
                         "costs, not whole-window 118-channel timings or a proposed padding default.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--formula-only", action="store_true",
                        help="Refresh DC formula checks without repeating factorization experiments or timings.")
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    if args.formula_only:
        path = ROOT / "audit/briefing/spectral_directed_followup.json"
        report = json.loads(path.read_text())
        report['scope'] = (
            "Initial investigation before the DC correction, preserved as a historical checkpoint. "
            "The directed_coherence_correction field records the targeted post-correction check; "
            "factorization experiments and timings were not repeated.")
        report['directed_coherence_correction'] = dict(
            formula_checks=formula_checks(),
            source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                           for name in ("pyspi/_spectral.py", "pyspi/statistics/spectral.py",
                                        str(Path(__file__).relative_to(ROOT)))})
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report['directed_coherence_correction'], indent=2))
        return
    with contextlib.redirect_stdout(io.StringIO()):
        raw = raw_fixture()
        formulas = formula_checks()
        ordering = order_checks(raw)
        padding = [padding_check(raw[:2, -1024:], m, tol, its)
                   for m, tol, its in [(1, 1e-8, 60), (1, 1e-8, 600),
                                        (1, 1e-12, 600), (2, 1e-8, 60), (4, 1e-8, 60)]]
        conditioning = conditioning_check()
        timings = timing_checks(raw[:2, -1024:], args.repeats)
    report = dict(
        scope="Squared directed coherence corrected and checked against analytic references. "
              "Orientation, factorization, padding and smoothing remain unchanged; those experiments are diagnostic.",
        seed=821, formula_checks=formulas, ordering=ordering, padding_experiments=padding,
        conditioning=conditioning, padding_timings=timings, cpu_affinity=sorted(os.sched_getaffinity(0)),
        environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "spectral-connectivity")},
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ("pyspi/_spectral.py", "pyspi/statistics/spectral.py", str(Path(__file__).relative_to(ROOT)))},
    )
    path = ROOT / "audit/briefing/spectral_directed_followup.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
