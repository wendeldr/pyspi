"""Spectral evaluation refactor: estimator agreement, sharing, and cache lifetime.

Dependency comparisons cover the estimators whose definitions are unchanged.
PSI's agreed definition is checked independently in test_psi.py.
"""

from unittest.mock import patch

import numpy as np
import pytest

from pyspi._cache import ArrayCache
from pyspi._spectral import SpectralConnectivity, data_spectral_cache, spectral_workspace
from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi.data import Data
from pyspi.statistics import spectral


SPECTRAL_CLASSES = [
    spectral.CoherenceMagnitude, spectral.CoherencePhase, spectral.ImaginaryCoherence,
    spectral.PhaseLockingValue, spectral.PhaseLagIndex, spectral.WeightedPhaseLagIndex,
    spectral.DebiasedSquaredPhaseLagIndex, spectral.DebiasedSquaredWeightedPhaseLagIndex,
    spectral.PairwisePhaseConsistency,
]
DIRECTED_CLASSES = [
    spectral.DirectedCoherence, spectral.PartialDirectedCoherence,
    spectral.GeneralizedPartialDirectedCoherence, spectral.DirectedTransferFunction,
    spectral.DirectDirectedTransferFunction,
]


def make_data(n=128, p=4, **kwargs):
    raw = np.random.default_rng(940).normal(size=(p, n))
    if p > 1:
        raw[1] = .8 * np.roll(raw[0], 3) + .2 * raw[1]
    return Data(raw, normalise=False, **kwargs)


def reference(data, fs=1, indices=None):
    raw = data.to_numpy(realisation=0)
    if indices is not None:
        raw = raw[indices]
    return sc.Connectivity.from_multitaper(sc.Multitaper(raw.T, sampling_frequency=fs))


@pytest.mark.parametrize("cls", SPECTRAL_CLASSES)
@pytest.mark.parametrize("n", [64, 129])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_spectral_estimator_and_band_agreement(cls, n, statistic):
    data = make_data(n)
    spi = cls(fmin=.02, fmax=.4, statistic=statistic)
    conn = reference(data)
    mask = (conn.frequencies >= .02) & (conn.frequencies <= .4)
    values = getattr(conn, spi.measure)()[0, mask]
    expected = getattr(np, "nan" + statistic)(values, axis=0)
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("cls", DIRECTED_CLASSES)
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_pairwise_models_match_separate_ordered_pair_reference(cls, statistic):
    data = make_data(p=3)
    spi = cls(fmin=.02, fmax=.4, statistic=statistic)
    expected = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            if i != j:
                conn = reference(data, indices=[i, j])
                mask = (conn.frequencies >= .02) & (conn.frequencies <= .4)
                values = getattr(conn, spi.measure)()[0, mask, 0, 1]
                if cls is spectral.DirectDirectedTransferFunction:
                    # dDTF normalizes over its entire one-sided frequency grid.
                    # Include the newly exposed Nyquist term in that denominator.
                    from spectral_connectivity.connectivity import _estimate_transfer_function
                    transfer = _estimate_transfer_function(conn._minimum_phase_factor)[0]
                    old_energy = np.sum(abs(transfer[:64, 0, :])**2)
                    full_energy = np.sum(abs(transfer[:65, 0, :])**2)
                    values *= np.sqrt(old_energy / full_energy)
                expected[i, j] = getattr(np, "nan" + statistic)(values)
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(spi.bivariate(data, i=2, j=0), expected[2, 0], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("cls,statistic", [
    (spectral.GroupDelay, "delay"), (spectral.GroupDelay, "slope"), (spectral.GroupDelay, "rvalue"),
])
def test_group_delay_preserves_fit_and_significance_with_transform_spacing(cls, statistic):
    data = make_data(n=512, p=3)
    spi = cls(fmin=.02, fmax=.4, statistic=statistic)
    conn = reference(data)
    expected = getattr(conn, spi.measure)(
        frequencies_of_interest=(.02, .4), frequency_resolution=6/512,
    )
    if cls is spectral.GroupDelay:
        expected = expected[("delay", "slope", "rvalue").index(statistic)]
    expected = expected[0].copy()
    np.fill_diagonal(expected, np.nan)
    assert np.isfinite(expected).any(), "Reference must exercise actual estimates, not only NaNs"
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    actual[:] = 123  # A returned adjacency must not expose a writable cached view.
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-10, atol=1e-12)


def test_nonparametric_granger_matches_dependency():
    data = make_data(p=3)
    conn = reference(data)
    spi = spectral.SpectralGrangerCausality(fmin=.02, fmax=.4)
    mask = (conn.frequencies >= .02) & (conn.frequencies <= .4)
    values = conn.pairwise_spectral_granger_prediction()[0, mask]
    expected = np.nanmean(values, axis=0)
    nan_pct = np.isnan(values).mean(axis=0)
    np.fill_diagonal(nan_pct, 0)
    expected[nan_pct > spi.nan_threshold] = np.nan
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("p", [3, 32, 118])
def test_exact_in_phase_and_antiphase_channels_keep_phase_branch(p):
    raw = np.random.default_rng(8).normal(size=(p, 64))
    raw[1], raw[2] = -raw[0], 2 * raw[0]
    data = Data(raw, normalise=False)
    coefficients = sc.Multitaper(raw.T, sampling_frequency=1).fft()[0, :, :, :33]
    cross = np.einsum("rkfi,rkfj->fij", coefficients, coefficients.conj(), optimize=False)
    expected = np.angle(cross).mean(axis=0)
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(spectral.CoherencePhase().multivariate(data), expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("cls", SPECTRAL_CLASSES)
def test_zero_channels_do_not_contaminate_other_pairs(cls):
    raw = make_data(p=3).to_numpy(realisation=0).copy()
    raw[2] = 0
    data = Data(raw, normalise=False)
    spi = cls()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = cls().multivariate(Data(raw[:2], normalise=False))
        actual = spi.multivariate(data)
    np.testing.assert_allclose(actual[:2, :2], expected, rtol=1e-11, atol=1e-13)


def test_transform_and_coherency_shared_across_measures_and_bands():
    data = make_data()
    with patch.object(sc.Multitaper, "fft", autospec=True, side_effect=sc.Multitaper.fft) as fft:
        with patch.object(SpectralConnectivity, "_mean_cross_spectrum", wraps=SpectralConnectivity._mean_cross_spectrum) as cross:
            for cls in (spectral.CoherenceMagnitude, spectral.CoherencePhase, spectral.PhaseSlopeIndex, spectral.GroupDelay):
                for lo, hi in ((0, .5), (0, .25), (.25, .5)):
                    kwargs = {"statistic": "delay"} if cls is spectral.GroupDelay else {}
                    cls(fmin=lo, fmax=hi, **kwargs).multivariate(data)
            assert fft.call_count == 1
            assert cross.call_count == 1


def test_pair_factorization_shared_across_measures_but_keeps_channel_order():
    import spectral_connectivity.connectivity as dependency
    data = make_data(p=3)
    with patch.object(dependency, "minimum_phase_decomposition", wraps=dependency.minimum_phase_decomposition) as factor:
        for cls in DIRECTED_CLASSES:
            cls().multivariate(data)
        spectral.SpectralGrangerCausality().multivariate(data)
        assert factor.call_count == 6  # six ordered pairs; Granger reuses three


def test_directed_band_reuse_survives_pair_cache_eviction():
    import spectral_connectivity.connectivity as dependency
    data = make_data(p=7)
    with patch.object(dependency, "minimum_phase_decomposition", wraps=dependency.minimum_phase_decomposition) as factor:
        for lo, hi in ((0, .5), (0, .25), (.25, .5)):
            spectral.DirectedCoherence(fmin=lo, fmax=hi).multivariate(data)
        assert factor.call_count == 42
        assert len(spectral_workspace(data, 1)._pairs) == 16


@pytest.mark.parametrize("mutation", ["replace", "add", "remove", "view"])
def test_changed_input_invalidates_all_spectral_work(mutation):
    data = make_data()
    spi = spectral.CoherenceMagnitude()
    spi.multivariate(data)
    old = spectral_workspace(data, 1)
    if mutation == "replace":
        data.set_data(np.random.default_rng(55).normal(size=(4, 128)))
    elif mutation == "add":
        data.add_process(np.random.default_rng(55).normal(size=128))
    elif mutation == "remove":
        data.remove_process(1)
    else:
        data.to_numpy()[0, :, 0] = np.random.default_rng(55).normal(size=128)
    expected = spi.multivariate(Data(data.to_numpy().copy(), dim_order="psr", normalise=False))
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-13)
    assert spectral_workspace(data, 1) is not old


def test_sampling_rates_are_isolated_and_frequency_units_scale():
    data = make_data()
    for fs in (1, 2, 128, 2, 1):
        spi = spectral.CoherenceMagnitude(fs=fs, fmin=.05*fs, fmax=.3*fs)
        expected = spi.multivariate(make_data())
        np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(spectral_workspace(data, fs).frequencies, np.fft.rfftfreq(128, 1/fs))
    assert len(data_spectral_cache(data).workspaces) == 2


@pytest.mark.parametrize("kind", ["unit", "sign", "imag", "abs_imag", "imag_squared"])
def test_phase_blocks_bound_temporaries_and_preserve_expectation(monkeypatch, kind):
    data = make_data(n=64, p=5)
    conn = spectral_workspace(data, 1)
    monkeypatch.setattr(conn, "block_bytes", 5 * 5 * 16 * 2)
    functions = {"unit": lambda x: x/abs(x), "sign": lambda x: np.sign(x.imag),
                 "imag": lambda x: x.imag, "abs_imag": lambda x: abs(x.imag),
                 "imag_squared": lambda x: x.imag**2}
    expected = reference(data)._expectation_cross_spectral_matrix(fcn=functions[kind])[:, :33]
    sizes = []
    multiply = np.matmul
    def record(a, b, **kwargs):
        result = multiply(a, b, **kwargs)
        sizes.append(result.nbytes)
        return result
    with patch("pyspi._spectral.np.matmul", side_effect=record):
        actual = conn._phase_statistic(kind)
    assert len(sizes) > 1
    assert max(sizes) <= conn.block_bytes
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_array_and_pair_caches_evict_within_limits():
    cache = ArrayCache(max_bytes=32)
    a = cache.get("a", lambda: np.arange(4, dtype=float))
    cache.get("b", lambda: np.arange(4, dtype=float))
    assert list(cache.entries) == ["b"] and cache.nbytes == 32
    cache.get("large", lambda: np.arange(5, dtype=float))
    assert list(cache.entries) == ["b"]
    np.testing.assert_array_equal(a, np.arange(4))
    conn = spectral_workspace(make_data(p=4), 1)
    conn.max_pairs = 2
    first = conn.pair(0, 1)
    assert conn.pair(0, 1) is first
    conn.pair(0, 2)
    conn.pair(0, 3)
    assert len(conn._pairs) == 2 and (0, 1) not in conn._pairs


@pytest.mark.parametrize("cls", [spectral.CoherenceMagnitude, spectral.DirectedCoherence])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_empty_frequency_band_is_unavailable(cls, statistic):
    assert np.isnan(cls(fmin=.001, fmax=.002, statistic=statistic).multivariate(make_data())).all()


@pytest.mark.parametrize("kwargs", [{"fs": 0}, {"fs": np.nan}, {"fmin": -.1}, {"fmin": .4, "fmax": .2}, {"fmax": .6}])
def test_invalid_frequency_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        spectral.CoherenceMagnitude(**kwargs)


def test_unsupported_input_axes_and_short_windows_are_explicit():
    data = Data(np.ones((2, 32, 2)), dim_order="psr", normalise=False)
    with pytest.raises(ValueError, match="single realisation"):
        spectral.CoherenceMagnitude().multivariate(data)
    with pytest.raises(ValueError, match="more than six"):
        spectral.CoherenceMagnitude().multivariate(make_data(n=6))
    with pytest.raises(ValueError, match="Both channel indices"):
        spectral.DirectedCoherence().bivariate(make_data())


def test_parametric_granger_sampling_rate_and_mutation():
    data = make_data(n=256, p=3)
    for fs in (1, 2):
        spi = spectral.SpectralGrangerCausality(method="parametric", order=2, fs=fs, fmin=.02*fs, fmax=.4*fs)
        values, freq = spi._get_cache(data)
        fresh_values, fresh_freq = spi._get_cache(make_data(n=256, p=3))
        np.testing.assert_allclose(values, fresh_values, equal_nan=True)
        np.testing.assert_array_equal(freq, fresh_freq)
        assert freq[-1] == fs/2
    data.to_numpy()[0, :, 0] *= -1
    fresh = Data(data.to_numpy().copy(), dim_order="psr", normalise=False)
    np.testing.assert_allclose(spi.multivariate(data), spi.multivariate(fresh), equal_nan=True)
