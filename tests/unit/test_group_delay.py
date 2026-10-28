"""Independent significance, selection, regression and diagnostic checks."""

import numpy as np
import pytest
from scipy.ndimage import label
from scipy.stats import beta, false_discovery_control, linregress

from pyspi._group_delay import REASONS, coherence_pvalues, frequency_bh
from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi.data import Data
from pyspi.statistics.spectral import GroupDelay


def injected(pair_values, *, nfft=64, resolution=2):
    values = np.atleast_2d(pair_values)
    conn = SpectralConnectivity(np.zeros((len(values), 1, 5, nfft, 2), complex),
                                np.fft.fftfreq(nfft, 1/nfft), np.arange(len(values)),
                                frequency_resolution=resolution)
    c = np.ones((len(values), values.shape[-1], 2, 2), complex)
    c[..., 0, 1], c[..., 1, 0] = values, values.conj()
    conn.__dict__["_coherency"] = c
    return conn


def scalar_reference(conn, band):
    """Independent beta/FDR/label/OLS implementation, one pair at a time."""
    f = conn.frequencies
    mask = (f >= band[0]) & (f <= band[1])
    c = conn.coherency()[:, mask]
    indices = np.flatnonzero(mask)
    nfft = conn.fourier_coefficients.shape[-2]
    real = (indices == 0) | ((nfft % 2 == 0) & (indices == nfft//2))
    step = int(np.ceil(conn.frequency_resolution / (f[1] - f[0])))
    f = f[mask]
    shape = (len(c), c.shape[-1], c.shape[-1])
    result = [np.full(shape, np.nan) for _ in range(3)]
    for w in range(len(c)):
        for i in range(c.shape[-1]):
            for j in range(i + 1, c.shape[-1]):
                squared = abs(c[w, :, i, j])**2
                p = beta.sf(squared, 1, conn.n_observations - 1)
                p[real] = beta.sf(squared[real], .5, (conn.n_observations - 1)/2)
                marked = false_discovery_control(p) <= .05
                groups, number = label(marked)
                if number == 0:
                    continue
                biggest = 1 + np.argmax([np.sum(groups == k) for k in range(1, number + 1)])
                run = np.flatnonzero(groups == biggest)
                keep = run[::step]
                if len(keep) < 3:
                    continue
                fit = linregress(f[keep], np.unwrap(np.angle(c[w, run, i, j]))[::step])
                for out, value in zip(result, (fit.slope/(2*np.pi), fit.slope, fit.rvalue)):
                    out[w, i, j], out[w, j, i] = value, -value
    return result


@pytest.mark.parametrize("nfft", [63, 64])
def test_pipeline_agrees_with_independent_reference(nfft):
    rng = np.random.default_rng(651)
    raw = rng.normal(size=(4, nfft))
    raw[1] = np.roll(raw[0], 2) + .15*raw[1]
    conn = spectral_workspace(Data(raw, normalise=False), nfft)
    band = (0, nfft/2)
    expected = scalar_reference(conn, band)
    assert np.isfinite(expected[0]).any()
    for actual, reference in zip(conn.group_delay(band), expected):
        np.testing.assert_allclose(actual, reference, atol=2e-13)


@pytest.mark.parametrize("real", [False, True])
def test_null_calibration_for_complex_and_real_observations(real):
    rng = np.random.default_rng(652)
    x = rng.normal(size=(30000, 5, 2))
    if not real:
        x = x + 1j*rng.normal(size=x.shape)
    c = np.sum(x[..., 0]*x[..., 1].conj(), axis=1) / np.sqrt(
        np.sum(abs(x[..., 0])**2, axis=1)*np.sum(abs(x[..., 1])**2, axis=1))
    p, invalid = coherence_pvalues(c[:, None], 5, np.array([real]))
    assert not invalid.any()
    assert .045 < np.mean(p <= .05) < .055


def test_frequency_bh_matches_scipy_independently_per_window_and_pair():
    p = np.random.default_rng(653).uniform(size=(3, 8, 71))
    p[:, ::2, :9] *= .0001
    p[0, 0, :] = .05  # BH includes equality and ties.
    np.testing.assert_array_equal(frequency_bh(p, .05), false_discovery_control(p, axis=-1) <= .05)
    np.testing.assert_array_equal(frequency_bh(p, .05)[0, 0], frequency_bh(p[:1, :1], .05)[0, 0])


def test_pair_estimates_unchanged_by_unrelated_channels_and_reordering():
    rng = np.random.default_rng(654)
    raw = rng.normal(size=(7, 1024))
    raw[1] = .6*np.roll(raw[0], 3) + .4*raw[1]
    spi = GroupDelay(fs=2048, fmin=2, fmax=250)
    pair = spi.multivariate(Data(raw[:2], normalise=False))
    all_channels = spi.multivariate(Data(raw, normalise=False))
    assert np.isfinite(pair[0, 1])
    np.testing.assert_allclose(pair, all_channels[:2, :2], atol=1e-14)
    order = [5, 1, 3, 0, 2, 6, 4]
    reordered = spi.multivariate(Data(raw[order], normalise=False))
    np.testing.assert_allclose(reordered, all_channels[np.ix_(order, order)], atol=1e-14)


def test_missingness_reasons_and_unwrap_are_local_to_selected_run():
    f = np.arange(33.)
    strong = .999*np.exp(.1j*f)
    values = np.stack([strong.copy() for _ in range(5)])
    values[0] = np.nan
    values[1] *= .1
    values[2] = .1
    values[2, 5:8] = strong[5:8]  # Only two points after spacing by two.
    values[3, :4] = np.nan        # An earlier NaN must not poison the later fit.
    values[4] = .65              # Old Fisher test accepted this; the zero-null does not.
    conn = injected(values)
    delay, _, r, reason, count, invalid = conn.group_delay_details((0, 32))
    assert [REASONS[k] for k in reason[:, 0, 1]] == [
        "invalid_coherency", "no_significant_frequencies", "insufficient_frequency_span",
        "estimated", "no_significant_frequencies"]
    assert count[2, 0, 1] == 2 and invalid[3, 0, 1] == 4
    np.testing.assert_allclose(delay[3, 0, 1], .1/(2*np.pi), atol=1e-15)
    assert r[3, 0, 1] == pytest.approx(1)
    # A window's significance is independent of other windows as well.
    isolated = injected(values[3]).group_delay((0, 32))[0]
    np.testing.assert_allclose(isolated[0], delay[3], atol=1e-15)


def test_first_longest_cluster_and_unwrap_before_decimation():
    f = np.arange(33.)
    values = np.full(33, .1, complex)
    values[1:8] = .999*np.exp(2j*f[1:8])
    values[15:22] = .999*np.exp(-.5j*f[15:22])
    conn = injected(values)
    result = conn.group_delay_details((0, 32))
    assert result[4][0, 0, 1] == 4
    np.testing.assert_allclose(result[0][0, 0, 1], 2/(2*np.pi), atol=1e-14)


def test_constant_phase_has_zero_delay_but_undefined_r():
    result = injected(np.full(33, .999+0j)).group_delay_details((0, 32))
    assert result[0][0, 0, 1] == 0
    assert np.isnan(result[2][0, 0, 1])
    assert REASONS[result[3][0, 0, 1]] == "estimated"


def test_unrepresentable_slope_is_reported_as_numerical_failure():
    from pyspi._group_delay import estimate
    c = injected(.999*np.exp(.1j*np.arange(33))).coherency()
    result = estimate(c, np.arange(33)*1e-310, 5, np.zeros(33, bool), 2, .05, 2**20)
    assert REASONS[result[3][0, 0, 1]] == "numerical_failure"
    assert all(np.isnan(x[0, 0, 1]) for x in result[:3])


def test_invalid_coherency_and_rounding_are_distinguished():
    c = np.array([[np.nan, np.inf, 1.1, 1+np.finfo(float).eps, 1, 0]], complex)
    p, invalid = coherence_pvalues(c, 5, np.zeros(6, bool))
    np.testing.assert_array_equal(invalid, [[True, True, True, False, False, False]])
    np.testing.assert_array_equal(p, [[1, 1, 1, 0, 0, 1]])
    with pytest.raises(ValueError, match="at least two Fourier observations"):
        coherence_pvalues(c, 1, np.zeros(6, bool))


def test_diagnostics_variants_cache_and_blocking(monkeypatch):
    raw = np.random.default_rng(655).normal(size=(3, 512))
    raw[1] = np.roll(raw[0], 3) + .1*raw[1]
    data = Data(raw, normalise=False)
    conn = spectral_workspace(data, 512)
    spi = GroupDelay(fs=512, fmin=2, fmax=200)
    expected = spi.multivariate(data)
    cached = spi.diagnostics(data)
    original = cached['reason'].copy()
    cached['reason'][:] = 'overwritten'
    np.testing.assert_array_equal(spi.diagnostics(data)['reason'], original)
    monkeypatch.setattr(conn, '_group_delay_details', lambda *args: pytest.fail('fit repeated'))
    for statistic, scale in [('delay', 1), ('slope', 2*np.pi)]:
        result = GroupDelay(fs=512, fmin=2, fmax=200, statistic=statistic).multivariate(data)
        np.testing.assert_allclose(result, expected*scale, atol=1e-12)
    # Force one-pair temporary blocks in a fresh workspace.
    other = spectral_workspace(Data(raw.copy(), normalise=False), 512)
    other.block_bytes = 1
    np.testing.assert_allclose(other.group_delay((2, 200))[0][0], expected, atol=1e-12)


@pytest.mark.parametrize('alpha', [0, 1, -1, np.nan])
def test_invalid_significance_threshold(alpha):
    with pytest.raises(ValueError, match='threshold'):
        injected(np.ones(33)).group_delay((0, 32), significance_threshold=alpha)
