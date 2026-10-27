"""Independent phase-lag definitions, debiasing preservation and shared work."""

from unittest.mock import patch

import numpy as np
import pytest

from pyspi._cache import ArrayCache
from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi.base import Directed, Signed, Undirected, Unsigned
from pyspi.data import Data
from pyspi.statistics import spectral


def connection(imaginary):
    imaginary = np.asarray(imaginary, float)
    n, frequencies = imaginary.shape
    coefficients = np.ones((1, 1, n, 2 * frequencies, 2), complex)
    coefficients[0, 0, :, 1:frequencies+1, 1] -= 1j * imaginary
    return SpectralConnectivity(coefficients, np.fft.fftfreq(2 * frequencies), np.array([0.]))


@pytest.mark.parametrize("ordinary,signed", [
    (spectral.PhaseLagIndex, spectral.SignedPhaseLagIndex),
    (spectral.WeightedPhaseLagIndex, spectral.SignedWeightedPhaseLagIndex),
])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_magnitude_is_taken_before_band_summary(ordinary, signed, statistic):
    # Per-frequency signed values +.6 and -.6. abs(mean) would incorrectly be 0.
    conn = connection([[1, -1], [1, -1], [1, -1], [1, -1], [-1, 1]])
    data = Data(np.ones((2, 8)), normalise=False)
    kwargs = dict(fmin=.25, fmax=.5, statistic=statistic)
    with patch("pyspi.statistics.spectral.spectral_workspace", return_value=conn):
        strength = ordinary(**kwargs).multivariate(data)
        sign = signed(**kwargs).multivariate(data)
    np.testing.assert_allclose(strength[0, 1], .6)
    np.testing.assert_allclose(strength, strength.T)
    np.testing.assert_allclose(sign[0, 1], 0 if statistic == "mean" else .6)
    assert isinstance(ordinary(), Unsigned) and isinstance(ordinary(), Undirected)
    assert isinstance(signed(), Signed) and not isinstance(signed(), Undirected)
    assert isinstance(signed(), Directed)
    assert ordinary().identifier != signed().identifier


def test_weighted_definition_and_debiased_pair_products():
    imaginary = np.array([[10, 1], [10, 1], [-1, 1], [-1, -1], [-1, -1.]])
    conn = connection(imaginary)
    n = len(imaginary)
    actual = {name: conn.measure(name)[0, 1:, 0, 1] for name in (
        "phase_lag_index", "signed_phase_lag_index", "weighted_phase_lag_index",
        "signed_weighted_phase_lag_index", "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index", "phase_locking_value",
        "pairwise_phase_consistency")}
    for f in range(imaginary.shape[1]):
        y = imaginary[:, f]
        signs = np.sign(y)
        unit = (1+1j*y) / np.abs(1+1j*y)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        expected = {
            "signed_phase_lag_index": signs.mean(),
            "phase_lag_index": abs(signs.mean()),
            "signed_weighted_phase_lag_index": y.sum()/abs(y).sum(),
            "weighted_phase_lag_index": abs(y.sum())/abs(y).sum(),
            "debiased_squared_phase_lag_index": np.mean([signs[i]*signs[j] for i, j in pairs]),
            "debiased_squared_weighted_phase_lag_index": (
                sum(y[i]*y[j] for i, j in pairs)/sum(abs(y[i]*y[j]) for i, j in pairs)),
            "phase_locking_value": abs(unit.mean()),
            "pairwise_phase_consistency": np.mean([(unit[i]*unit[j].conjugate()).real
                                                    for i, j in pairs]),
        }
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name][f], value, atol=1e-14)
    assert actual["debiased_squared_phase_lag_index"][1] == pytest.approx(-.2)
    assert actual["debiased_squared_weighted_phase_lag_index"][1] == pytest.approx(-.2)


@pytest.mark.parametrize("name", ["debiased_squared_phase_lag_index",
                                 "debiased_squared_weighted_phase_lag_index",
                                 "pairwise_phase_consistency"])
def test_negative_debiased_estimates_are_preserved_and_symmetric(name):
    # Alternating quadrature phases: correction produces a finite negative estimate.
    coeff = np.ones((1, 1, 4, 8, 2), complex)
    coeff[0, 0, :, :, 1] = np.array([1j, -1j, 1j, -1j])[:, None]
    conn = SpectralConnectivity(coeff, np.fft.fftfreq(8), np.array([0.]))
    value = conn.measure(name)
    np.testing.assert_allclose(value[0, :, 0, 1], -1/3)
    np.testing.assert_allclose(value[0, :, 1, 0], -1/3)


@pytest.mark.parametrize("cls", [spectral.DebiasedSquaredPhaseLagIndex,
                                 spectral.DebiasedSquaredWeightedPhaseLagIndex,
                                 spectral.PairwisePhaseConsistency])
def test_debiased_metadata_describes_negative_values_without_direction(cls):
    spi = cls()
    assert spi.issigned() and "signed" in spi.labels and "unsigned" not in spi.labels
    assert isinstance(spi, Undirected) and "undirected" in spi.labels


def test_weighted_moments_share_cross_products_and_reuse_cached_moments():
    conn = spectral_workspace(Data(np.random.default_rng(42).normal(size=(4, 64))), 1)
    with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduce:
        conn.measure("signed_weighted_phase_lag_index")
        conn.measure("weighted_phase_lag_index")
        conn.measure("debiased_squared_weighted_phase_lag_index")
        assert [call.args[0] for call in reduce.call_args_list] == [
            ["imag", "abs_imag"], ["imag_squared"]]


def test_multimoment_cross_products_are_bounded_and_formed_only_once(monkeypatch):
    conn = spectral_workspace(Data(np.random.default_rng(41).normal(size=(5, 64))), 1)
    monkeypatch.setattr(conn, "block_bytes", 5*5*16*2)
    multiply = np.matmul
    sizes = []
    def record(a, b, **kwargs):
        value = multiply(a, b, **kwargs)
        sizes.append(value.nbytes)
        return value
    with patch("pyspi._spectral.np.matmul", side_effect=record):
        conn.measure("debiased_squared_weighted_phase_lag_index")
    joint_sizes = sizes.copy()
    sizes.clear()
    with patch("pyspi._spectral.np.matmul", side_effect=record):
        conn._reduce_phase(["sign"])
    assert joint_sizes == sizes and max(sizes) <= conn.block_bytes


def test_batched_cache_preserves_requested_hits_during_eviction():
    cache = ArrayCache(max_bytes=32)
    first = cache.get("first", lambda: np.arange(4.))
    values = cache.get_many(["first", "second", "third"],
                            lambda missing: {key: np.ones(4) for key in missing})
    assert values["first"] is first
    assert cache.nbytes == 32 and list(cache.entries) == ["third"]
    assert len(values) == 3


def test_signed_unsigned_and_debiased_variants_share_signs_and_are_order_independent():
    raw = np.random.default_rng(12).normal(size=(3, 64))
    classes = [spectral.PhaseLagIndex, spectral.SignedPhaseLagIndex,
               spectral.DebiasedSquaredPhaseLagIndex]
    data = Data(raw)
    conn = spectral_workspace(data, 1)
    with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduce:
        actual = [cls().multivariate(data) for cls in classes]
        assert reduce.call_count == 1
    fresh = Data(raw)
    for cls, expected in reversed(list(zip(classes, actual))):
        np.testing.assert_allclose(cls().multivariate(fresh), expected)
