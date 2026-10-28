"""Nondegenerate input/aggregation checks supplement single-dataset snapshots.

These freeze wrapper behavior, including current band endpoints and triangle
orientation. They do not validate the dependency's connectivity estimators.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.signal import detrend
from scipy.stats import zscore

from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics import basic, spectral, wavelet
from regression import harness, spi_policy


@pytest.mark.parametrize("shape", [(2, 32), (5, 137), (7, 1024)])
@pytest.mark.parametrize("squared", [False, True])
def test_covariance_and_precision_match_independent_matrix_products(shape, squared):
    raw = np.random.default_rng(38).normal(size=shape)
    raw[1] += 0.7 * raw[0]
    raw *= np.arange(1, shape[0] + 1)[:, None]
    raw += np.arange(shape[0])[:, None]
    centered = raw - raw.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / shape[1]
    precision = np.linalg.inv(covariance)
    data = Data(raw, normalise=False, detrend=False)
    # Both statistics and their squared versions reuse a fitted estimator.
    for cls, reference in ((basic.Covariance, covariance), (basic.Precision, precision)):
        expected = reference ** 2 if squared else reference.copy()
        np.fill_diagonal(expected, np.nan)
        harness.assert_spi_allclose(cls(squared=squared).multivariate(data), expected, cls.__name__,
                                   rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("order", ["psr", "rsp", "spr"])
@pytest.mark.parametrize("normalise,detrended", [(False, False), (True, False), (False, True), (True, True)])
def test_preprocessing_replications_preserves_axes(order, normalise, detrended):
    rng = np.random.default_rng(39)
    raw = rng.normal(size=(3, 64, 2)) * np.array([1, 3, 7])[:, None, None]
    raw += np.arange(64)[None, :, None] * np.array([0.3, -0.5, 2])[:, None, None]
    source = raw.transpose(tuple("psr".index(axis) for axis in order))
    expected = detrend(raw, axis=1) if detrended else raw.copy()
    if normalise:
        expected = zscore(expected, axis=1, ddof=1)
    data = Data(source, dim_order=order, normalise=normalise, detrend=detrended)
    assert (data.n_processes, data.n_observations, data.n_replications) == (3, 64, 2)
    np.testing.assert_allclose(data.to_numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("cls_name", ["CoherenceMagnitude", "ImaginaryCoherence",
    "PhaseLockingValue", "PhaseLagIndex", "WeightedPhaseLagIndex", "DebiasedSquaredPhaseLagIndex",
    "DebiasedSquaredWeightedPhaseLagIndex", "PairwisePhaseConsistency",
    "SignedPhaseLagIndex", "SignedWeightedPhaseLagIndex"])
def test_multitaper_aggregation_uses_selected_frequencies_and_preserves_orientation(monkeypatch, cls_name, statistic):
    cls = getattr(spectral, cls_name)
    spi = cls(fmin=0.1, fmax=0.3, statistic=statistic)
    frequencies = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
    tensor = np.empty((1, 5, 3, 3), dtype=float)
    offsets = [-20, -2, 1, 8, 40]
    for f, offset in enumerate(offsets):
        for row in range(3):
            for col in range(3):
                value = 3 * row - 2 * col + offset
                # The actual PLV backend returns magnitudes, not complex phase
                # vectors. Its magnitude reduction is checked independently.
                tensor[0, f, row, col] = abs(value)/64 if cls_name == "PhaseLockingValue" else value
    data = Data(np.arange(90.).reshape(3, 30), normalise=False)
    monkeypatch.setattr(spi, "_get_cache", lambda data: (tensor, frequencies))
    expected = np.full((3, 3), np.nan)
    for row in range(3):
        for col in range(3):
            if row == col:
                continue
            values = [tensor[0, f, row, col] for f in (1, 2, 3)]
            if cls_name in ("PhaseLagIndex", "WeightedPhaseLagIndex"):
                values = [abs(value) for value in values]
            expected[row, col] = sum(values) / 3 if statistic == "mean" else max(values)
    before = tensor.copy()
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(tensor, before)


@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("cls_name", ["DirectedCoherence", "PartialDirectedCoherence",
    "GeneralizedPartialDirectedCoherence", "DirectedTransferFunction", "DirectDirectedTransferFunction"])
def test_bivariate_spectral_aggregation_uses_inclusive_band_and_source_target(monkeypatch, cls_name, statistic):
    from pyspi._spectral import SpectralConnectivity, canonical_directed_measure

    spi = getattr(spectral, cls_name)(fmin=0.1, fmax=0.5, statistic=statistic)
    data = Data(np.arange(90.).reshape(3, 30), normalise=False)
    frequencies = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
    pairs = {}
    expected = np.full((3, 3), np.nan)
    for row in range(3):
        for col in range(3):
            if row == col:
                continue
            values = 10 * row + col + np.array([100., 200., -2., 8., 500.])
            key = (row, col)
            tensor = pairs.setdefault(key, np.full((1, 5, 2, 2), 9999.0))
            tensor[0, :, 0, 1] = values
            expected[row, col] = values[1:].mean() if statistic == "mean" else values[1:].max()
    # The wrapper streams unique pair fits into real band summaries. Stub
    # only those expensive fits/values, not the aggregation being tested.
    conn = SpectralConnectivity(np.zeros((1, 1, 1, 8, 3), complex),
        np.r_[frequencies, [-.3, -.2, -.1]], np.array([0.]))
    unique_pairs = np.array([(0, 1), (0, 2), (1, 2)])
    directed_values = np.stack([np.stack((pairs[i, j][0, :, 0, 1],
                                         pairs[j, i][0, :, 0, 1]), axis=-1)
                                for i, j in unique_pairs])
    conn.directed_requests = {(canonical_directed_measure(spi.measure), (.1, .5))}
    monkeypatch.setattr(conn, "_directed_blocks", lambda: iter([(unique_pairs, None)]))
    monkeypatch.setattr(conn, "_directed_values", lambda model, name: directed_values)
    monkeypatch.setattr(spectral, "directed_workspace", lambda data, fs: conn)
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-12)
    for i, j in ((0, 1), (1, 0), (2, 1)):
        assert spi.bivariate(data, i=i, j=j) == pytest.approx(expected[i, j])


@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("cls_name", ["CoherenceMagnitude", "CoherencePhase", "ImaginaryCoherence",
    "PhaseLockingValue", "PhaseLagIndex", "PairwisePhaseConsistency",
    "DebiasedSquaredWeightedPhaseLagIndex", "weighted_PhaseLagIndex", "debiased_weighted_squared_PhaseLagIndex"])
def test_wavelet_nondegenerate_aggregation_and_backend_cache(monkeypatch, cls_name, statistic):
    spi = getattr(wavelet, cls_name)(fmin=0.2, fmax=0.4, statistic=statistic)
    frequencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    phases = np.empty((3, 3, 5, 4))
    for row in range(3):
        for col in range(3):
            phases[row, col] = (0.2 * row - 0.3 * col + np.array([-0.8, -0.2, 0.3, 0.9, 1.3])[:, None]
                                + np.array([-0.2, 0.1, 0.4, 0.8])[None, :])
    tensor = 2 * np.exp(1j * phases) if cls_name == "CoherencePhase" else phases
    calls = []
    def backend(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(freqs=frequencies, get_data=lambda output: tensor)
    monkeypatch.setattr(wavelet.mnec, "spectral_connectivity_epochs", backend)
    raw = np.random.default_rng(40).normal(size=(3, 64, 2))
    data = Data(raw, dim_order="psr", normalise=False)
    expected = np.full((3, 3), np.nan)
    for row in range(3):
        for col in range(row):
            values = [phases[row, col, f, t] for f in (1, 2, 3) for t in range(4)]
            value = sum(values) / len(values) if statistic == "mean" else max(values)
            expected[row, col] = expected[col, row] = value
    for _ in range(2):
        np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-12)
    assert len(calls) == 1
    assert calls[0]["method"] == spi.measure
    np.testing.assert_array_equal(calls[0]["data"], raw.transpose(2, 0, 1))


def test_calculator_reload_recomputes_instead_of_reusing_previous_caches(tmp_path):
    path = tmp_path / "covariance.yaml"
    path.write_text(yaml.safe_dump({".statistics.basic": {"Covariance": {
        "configs": [{"estimator": "EmpiricalCovariance"}]}}}))
    calc = Calculator(configfile=str(path), normalise=False)
    previous = None
    for seed in (41, 42):
        raw = np.random.default_rng(seed).normal(size=(3, 80))
        raw[1] += (seed - 40) * raw[0]
        calc.load_dataset(raw)
        calc.compute()
        centered = raw - raw.mean(axis=1, keepdims=True)
        expected = centered @ centered.T / 80
        np.fill_diagonal(expected, np.nan)
        actual = calc.table["cov_empirical"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        if previous is not None:
            assert not np.allclose(actual, previous, equal_nan=True)
        previous = actual.copy()


def test_stochastic_spis_with_shared_caches_match_per_spi_seeded_goldens(
        golden_meta, golden_values, raw_dataset, seed_all, monkeypatch, tmp_path):
    config = {}
    for entry, _ in harness.iter_config_entries():
        if spi_policy.is_stochastic(entry["identifier"]):
            spec = config.setdefault(entry["module"], {}).setdefault(entry["cls"],
                {"configs": [], "dependencies": entry["deps"]})
            spec["configs"].append(entry["params"])
    path = tmp_path / "stochastic.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    calc = Calculator(dataset=harness.make_data(raw_dataset, golden_meta["dataset"]), configfile=str(path))
    assert calc.spis
    for spi in calc.spis.values():
        compute = spi.multivariate
        def seeded(data, compute=compute):
            seed_all(golden_meta["seed"])
            return compute(data)
        monkeypatch.setattr(spi, "multivariate", seeded)
    calc.compute()
    for ident in calc.spis:
        harness.assert_spi_allclose(calc.table[ident].to_numpy(), golden_values[ident], ident)
