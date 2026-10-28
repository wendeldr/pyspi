"""Independent direction and metadata checks for Fourier spectral measures."""

import numpy as np
import pytest

from pyspi._spectral import SpectralConnectivity
from pyspi.base import Directed, Signed, Undirected, Unsigned
from pyspi.data import Data
from pyspi.statistics import spectral


@pytest.mark.parametrize("cls", [spectral.PhaseSlopeIndex, spectral.GroupDelay,
                                 spectral.CoherencePhase])
def test_signed_ordered_metrics_have_matching_traits_and_labels(cls):
    spi = cls()
    assert spi.issigned() and isinstance(spi, Signed)
    assert isinstance(spi, Directed) and not isinstance(spi, Undirected)
    assert not isinstance(spi, Unsigned)
    assert "signed" in spi.labels and "unsigned" not in spi.labels
    assert "directed" in spi.labels and "undirected" not in spi.labels


@pytest.mark.parametrize("cls", [spectral.CoherenceMagnitude, spectral.ImaginaryCoherence,
                                 spectral.PhaseLockingValue, spectral.DirectedCoherence,
                                 spectral.PartialDirectedCoherence,
                                 spectral.GeneralizedPartialDirectedCoherence,
                                 spectral.DirectedTransferFunction,
                                 spectral.DirectDirectedTransferFunction,
                                 spectral.SpectralGrangerCausality])
def test_other_metrics_keep_their_unsigned_trait(cls):
    spi = cls()
    assert not spi.issigned() and isinstance(spi, Unsigned)


def exact_phase_connection(order=(0, 1, 2)):
    frequencies = np.arange(33.)
    phase = np.array([0*frequencies, -.02*frequencies-.0003*frequencies**2,
                      -.07*frequencies-.0004*frequencies**2])[list(order)]
    spectrum = .99 * np.exp(1j*(phase.T[:, :, None] - phase.T[:, None, :]))
    conn = SpectralConnectivity(np.zeros((1, 1, 5, 64, 3), complex),
                                np.fft.fftfreq(64, 1/64), np.array([0.]),
                                frequency_resolution=6)
    conn.__dict__["_coherency"] = spectrum[None]
    return conn, frequencies, phase


def test_group_delay_signed_r_is_pearson_correlation_for_each_ordered_pair():
    conn, frequencies, phase = exact_phase_connection()
    delay, slope, correlation = conn.group_delay((1, 24))
    order = [2, 0, 1]
    permuted = exact_phase_connection(order)[0].group_delay((1, 24))
    selected = (frequencies >= 1) & (frequencies <= 24)
    f = frequencies[selected][::6]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            y = (phase[i] - phase[j])[selected][::6]
            expected_slope = np.dot(f-f.mean(), y-y.mean()) / np.dot(f-f.mean(), f-f.mean())
            expected_r = np.corrcoef(f, y)[0, 1]
            np.testing.assert_allclose(slope[0, i, j], expected_slope, atol=1e-14)
            np.testing.assert_allclose(delay[0, i, j], expected_slope/(2*np.pi), atol=1e-14)
            np.testing.assert_allclose(correlation[0, i, j], expected_r, atol=1e-14)
    for before, after in zip((delay, slope, correlation), permuted):
        np.testing.assert_allclose(after[0], before[0][np.ix_(order, order)], atol=1e-14)


@pytest.mark.parametrize("statistic", ["delay", "slope", "rvalue"])
def test_group_delay_known_delay_sign_and_channel_permutation(statistic):
    source = np.random.default_rng(822).normal(size=2060)
    raw = np.stack([source[12:], source[9:-3], source[:-12]])
    spi = spectral.GroupDelay(statistic=statistic, fs=2048, fmin=2, fmax=400)
    a = spi.multivariate(Data(raw, normalise=False, detrend=False))
    assert np.all(a[np.triu_indices(3, 1)] > 0)
    np.testing.assert_allclose(a, -a.T, atol=1e-12)
    if statistic == "delay":
        np.testing.assert_allclose(a[0, 1], 3/2048, atol=.1/2048)
    order = [2, 0, 1]
    b = spi.multivariate(Data(raw[order], normalise=False, detrend=False))
    np.testing.assert_allclose(b, a[np.ix_(order, order)], atol=1e-12)


def test_group_delay_default_and_invalid_summaries_are_explicit():
    assert spectral.GroupDelay().identifier == spectral.GroupDelay(statistic="delay").identifier
    for statistic in ("mean", "max", "typo"):
        with pytest.raises(ValueError, match="delay, slope, or rvalue"):
            spectral.GroupDelay(statistic=statistic)


@pytest.fixture(scope="module")
def one_way_coupling():
    rng = np.random.default_rng(821)
    raw = np.zeros((3, 8192))
    noise = rng.normal(size=raw.shape)
    for t in range(1, raw.shape[1]):
        raw[0, t] = .5*raw[0, t-1] + noise[0, t]
        raw[1, t] = .2*raw[1, t-1] + .7*raw[0, t-1] + noise[1, t]
        raw[2, t] = noise[2, t]
    return Data(raw[:, -2048:], normalise=False, detrend=False)


@pytest.mark.parametrize("cls", [spectral.DirectedCoherence,
                                 spectral.PartialDirectedCoherence,
                                 spectral.GeneralizedPartialDirectedCoherence,
                                 spectral.DirectedTransferFunction,
                                 spectral.DirectDirectedTransferFunction,
                                 spectral.SpectralGrangerCausality])
def test_directed_source_first_orientation_and_channel_permutation(cls, one_way_coupling):
    # Channel 0 drives channel 1. This checks direction and permutation, not
    # statistical significance or the estimator's null bias.
    result = cls(fmin=.01, fmax=.2).multivariate(one_way_coupling)
    assert np.isfinite(result[1, 0]) and np.isfinite(result[0, 1])
    assert result[0, 1] > 3*result[1, 0]
    order = [2, 1, 0]
    permuted = Data(one_way_coupling.to_numpy(realisation=0)[order], normalise=False)
    actual = cls(fmin=.01, fmax=.2).multivariate(permuted)
    np.testing.assert_allclose(actual, result[np.ix_(order, order)], rtol=1e-10, atol=1e-11)


def test_parametric_granger_source_first_orientation_and_channel_permutation(one_way_coupling):
    spi = spectral.SpectralGrangerCausality(method="parametric", order=1, fmin=.01, fmax=.2)
    result = spi.multivariate(one_way_coupling)
    assert result[0, 1] > 3*result[1, 0]
    order = [2, 0, 1]
    permuted = Data(one_way_coupling.to_numpy(realisation=0)[order], normalise=False)
    np.testing.assert_allclose(spi.multivariate(permuted), result[np.ix_(order, order)], atol=1e-11)
