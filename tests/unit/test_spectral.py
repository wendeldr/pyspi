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
    spectral.SignedPhaseLagIndex, spectral.SignedWeightedPhaseLagIndex,
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


@pytest.mark.parametrize("cls", [cls for cls in SPECTRAL_CLASSES if cls is not spectral.CoherencePhase])
@pytest.mark.parametrize("n", [64, 129])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_spectral_estimator_and_band_agreement(cls, n, statistic):
    data = make_data(n)
    spi = cls(fmin=.02, fmax=.4, statistic=statistic)
    conn = reference(data)
    mask = (conn.frequencies >= .02) & (conn.frequencies <= .4)
    values = getattr(conn, spi.measure.removeprefix("signed_"))()[0, mask]
    if cls in (spectral.PhaseLagIndex, spectral.WeightedPhaseLagIndex):
        values = np.abs(values)
    expected = getattr(np, "nan" + statistic)(values, axis=0)
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("cls", DIRECTED_CLASSES)
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_pairwise_wrappers_match_individual_pair_formula_summaries(cls, statistic):
    data = make_data(p=3)
    spi = cls(fmin=.02, fmax=.4, statistic=statistic)
    expected = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            if i != j:
                pair_data = Data(data.to_numpy(realisation=0)[[i, j]], normalise=False)
                conn = spectral_workspace(pair_data, 1)
                mask = (conn.frequencies >= .02) & (conn.frequencies <= .4)
                # Formula/order integration reference on separately fitted pairs.
                # The factorizer itself has independent known-model tests.
                transfer = conn._transfer_function[0]
                power = abs(transfer)**2
                weights = power.copy()
                if cls in (spectral.DirectedCoherence, spectral.GeneralizedPartialDirectedCoherence):
                    weights *= np.diag(conn._noise_covariance[0])[None, None, :]
                values = (weights/weights.sum(axis=-1, keepdims=True))[:, 1, 0]
                if cls is spectral.DirectDirectedTransferFunction:
                    spectrum = transfer @ conn._noise_covariance[0] @ transfer.swapaxes(-1, -2).conj()
                    partial = abs(spectrum[:, 1, 0])**2 / (spectrum[:, 0, 0].real*spectrum[:, 1, 1].real)
                    values = power[:, 1, 0]/power[:, 1, :].sum()*partial
                expected[i, j] = getattr(np, "nan" + statistic)(values[mask])
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(spi.bivariate(data, i=2, j=0), expected[2, 0], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("cls,statistic", [
    (spectral.GroupDelay, "delay"), (spectral.GroupDelay, "slope"), (spectral.GroupDelay, "rvalue"),
])
def test_group_delay_summary_variants_and_cache_ownership(cls, statistic):
    data = make_data(n=512, p=3)
    spi = cls(fmin=.02, fmax=.4, statistic=statistic)
    conn = spectral_workspace(data, 1)
    # Mathematical references live in test_group_delay.py. This covers the
    # public wrappers selecting/copying each of the shared outputs.
    expected = conn.group_delay((.02, .4))[("delay", "slope", "rvalue").index(statistic)]
    expected = expected[0].copy()
    np.fill_diagonal(expected, np.nan)
    assert np.isfinite(expected).any(), "Reference must exercise actual estimates, not only NaNs"
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    actual[:] = 123  # A returned adjacency must not expose a writable cached view.
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-10, atol=1e-12)


def test_nonparametric_granger_pairwise_wrapper_matches_power_decomposition():
    data = make_data(p=3)
    spi = spectral.SpectralGrangerCausality(fmin=.02, fmax=.4)
    expected = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            conn = spectral_workspace(Data(data.to_numpy(realisation=0)[[i, j]], normalise=False), 1)
            h, covariance = conn._transfer_function[0], conn._noise_covariance[0]
            spectrum = h @ covariance @ h.swapaxes(-1, -2).conj()
            residual = covariance[0, 0]-covariance[0, 1]**2/covariance[1, 1]
            total = spectrum[:, 1, 1].real
            values = np.log(total/(total-residual*abs(h[:, 1, 0])**2))
            mask = (conn.frequencies >= .02) & (conn.frequencies <= .4)
            expected[i, j] = values[mask].mean()
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("p", [3, 32, 118])
def test_exact_in_phase_and_antiphase_channels_keep_phase_branch(p):
    raw = np.random.default_rng(8).normal(size=(p, 64))
    raw[1], raw[2] = -raw[0], 2 * raw[0]
    data = Data(raw, normalise=False)
    coefficients = sc.Multitaper(raw.T, sampling_frequency=1).fft()[0, :, :, :33]
    cross = np.einsum("rkfi,rkfj->fij", coefficients, coefficients.conj(), optimize=False)
    expected = np.angle((cross / abs(cross)).mean(axis=0))
    np.fill_diagonal(expected, np.nan)
    actual = spectral.CoherencePhase().multivariate(data)
    # +pi and -pi describe the same direction.
    np.testing.assert_allclose(np.exp(1j*actual), np.exp(1j*expected), rtol=1e-12, atol=1e-13)


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


def test_pair_factorization_shared_across_measures_and_both_directions():
    import pyspi._spectral as implementation
    data = make_data(p=3)
    with patch.object(implementation, "factorize", wraps=implementation.factorize) as factor:
        for cls in DIRECTED_CLASSES:
            cls().multivariate(data)
        spectral.SpectralGrangerCausality().multivariate(data)
        assert factor.call_count == 1
        assert factor.call_args.args[0].shape[0] == 3  # three unique pairs, batched
    conn = spectral_workspace(data, 1)
    assert conn.pair(0, 1) is conn.pair(1, 0)


def test_planned_directed_bands_share_bounded_streaming_pair_fits():
    import pyspi._spectral as implementation
    data = make_data(p=7)
    spis = [cls(fmin=lo, fmax=hi) for cls in DIRECTED_CLASSES
            for lo, hi in ((0, .5), (0, .25), (.25, .5))]
    for spi in spis:
        spi._register_shared_work(data)
    conn = spectral_workspace(data, 1)
    conn.block_bytes = 128*10*16*2  # two pairs per block
    with patch.object(implementation, "factorize", wraps=implementation.factorize) as factor:
        for spi in spis:
            spi.multivariate(data)
        assert sum(call.args[0].shape[0] for call in factor.call_args_list) == 21
        assert all(call.args[0].shape[0] <= 2 for call in factor.call_args_list)
        assert not conn._pairs  # no retained full pair models in the streaming path


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


@pytest.mark.parametrize("kind", ["unit", "sign", "imag", "abs_imag"])
def test_phase_blocks_bound_temporaries_and_preserve_expectation(monkeypatch, kind):
    data = make_data(n=64, p=5)
    conn = spectral_workspace(data, 1)
    monkeypatch.setattr(conn, "block_bytes", 5 * 5 * 16 * 2)
    functions = {"unit": lambda x: x/abs(x), "sign": lambda x: np.sign(x.imag),
                 "imag": lambda x: x.imag, "abs_imag": lambda x: abs(x.imag)}
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
