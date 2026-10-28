"""Envelope definitions, independent scalar references and numerical safeguards."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.fftpack import next_fast_len

from pyspi import _envelope as implementation
from pyspi._envelope import EnvelopeWorkspace, envelope_workspace
from pyspi.data import Data
from pyspi.statistics.misc import PowerEnvelopeCorrelation


SPECS = [(False, False, False), (False, True, False), (True, False, False),
         (True, False, True), (True, True, False), (True, True, True)]


def analytic_reference(raw):
    nfft = next_fast_len(raw.shape[-1])
    weight = np.zeros(nfft)
    weight[0] = 1
    weight[1:(nfft+1)//2] = 2
    if nfft % 2 == 0:
        weight[nfft//2] = 1
    return np.fft.ifft(np.fft.fft(raw, n=nfft)*weight)[..., :raw.shape[-1]]


def correlation(x, y):
    x, y = x-x.mean(), y-y.mean()
    return np.dot(x, y)/np.sqrt(np.dot(x, x)*np.dot(y, y))


def scalar_reference(raw, orth, log, absolute):
    z = analytic_reference(raw)
    amplitude = abs(z)
    transform = (lambda x: np.log(x**2)) if log else (lambda x: x)
    result = np.full((len(raw), len(raw)), np.nan)
    for i in range(len(raw)):
        for j in range(i+1, len(raw)):
            if orth:
                first = correlation(transform(amplitude[i]), transform(abs((z[j]*z[i].conj()/amplitude[i]).imag)))
                second = correlation(transform(amplitude[j]), transform(abs((z[i]*z[j].conj()/amplitude[j]).imag)))
                value = (abs(first)+abs(second))/2 if absolute else (first+second)/2
            else:
                value = correlation(transform(amplitude[i]), transform(amplitude[j]))
            result[i, j] = result[j, i] = value
    return result


@pytest.mark.parametrize("n", [64, 129, 1024])
@pytest.mark.parametrize("orth,log,absolute", SPECS)
def test_scalar_definitions_and_shared_variants(n, orth, log, absolute):
    raw = np.random.default_rng(n).normal(size=(4, n))
    spi = PowerEnvelopeCorrelation(orth=orth, log=log, absolute=absolute)
    actual = spi.multivariate(Data(raw, normalise=False))
    np.testing.assert_allclose(actual, scalar_reference(raw, orth, log, absolute), atol=2e-12, rtol=1e-10)
    np.testing.assert_array_equal(actual, actual.T)


def test_absolute_is_per_direction_before_averaging():
    work = EnvelopeWorkspace(np.zeros((2, 16)))
    directional = np.array([[np.nan, -.9], [.2, np.nan]])
    work._orthogonal = {False: (directional, np.array([[7, 0], [0, 7]]))}
    assert work.result(True, False, False)[0, 1] == pytest.approx(-.35)
    assert work.result(True, False, True)[0, 1] == pytest.approx(.55)


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("factor", [1., 2., -3., 1e-120, 1e120])
def test_proportional_orthogonal_signal_is_unavailable(log, factor):
    raw = np.random.default_rng(2).normal(size=(3, 256))
    raw[1] = factor*raw[0]
    data = Data(raw, normalise=False)
    spi = PowerEnvelopeCorrelation(orth=True, log=log)
    actual = spi.multivariate(data)
    assert np.isnan(actual[0, 1]) and np.isfinite(actual[0, 2])
    assert spi.diagnostics(data)["reason"][0, 1] == "degenerate_orthogonal_signal"


@pytest.mark.parametrize("orth,log,absolute", SPECS)
def test_channel_units_and_sign_do_not_change_envelope_correlation(orth, log, absolute):
    raw = np.random.default_rng(3).normal(size=(3, 256))
    spi = PowerEnvelopeCorrelation(orth=orth, log=log, absolute=absolute)
    expected = spi.multivariate(Data(raw, normalise=False))
    changed = raw*np.array([1e-140, -1e140, 7.])[:, None]
    np.testing.assert_allclose(spi.multivariate(Data(changed, normalise=False)), expected, atol=1e-12)


@pytest.mark.parametrize("log", [False, True])
def test_constant_envelope_is_not_a_large_arbitrary_correlation(log):
    t = np.arange(1024)
    raw = np.stack([np.cos(2*np.pi*7*t/1024), np.sin(2*np.pi*11*t/1024),
                    np.random.default_rng(4).normal(size=1024)])
    data = Data(raw, normalise=False)
    for orth in (False, True):
        spi = PowerEnvelopeCorrelation(orth=orth, log=log)
        assert np.isnan(spi.multivariate(data)).all()
        assert spi.diagnostics(data)["reason"][0, 2] == "constant_envelope"


def test_small_but_resolved_envelope_variation_is_valid_and_centered():
    t = np.arange(1024)
    phase = 2*np.pi*40*t/1024
    envelope = 1+1e-7*np.cos(2*np.pi*3*t/1024)
    other = 1+1e-7*np.sin(2*np.pi*3*t/1024)
    work = EnvelopeWorkspace(np.stack([envelope*np.exp(1j*phase), other*np.exp(1j*phase)]))
    for log in (False, True):
        result = work.result(False, log, False)
        assert np.isfinite(result[0, 1]) and abs(result[0, 1]) < 1e-8
        assert abs(result[0, 1]) <= 1


@pytest.mark.parametrize("log", [False, True])
def test_nearly_proportional_but_resolved_orthogonal_component_is_kept(log):
    raw = np.random.default_rng(71).normal(size=(2, 1024))
    raw[1] = raw[0]+1e-8*raw[1]
    work = EnvelopeWorkspace(raw)
    value = work.result(True, log, False)[0, 1]
    assert np.isfinite(value) and abs(value) <= 1
    assert work.diagnostics(True, log)["reason"][0, 1] == "estimated"


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_log_power_never_squares_unrepresentable_raw_amplitudes(scale):
    raw = np.random.default_rng(72).normal(size=(2, 256))
    expected = EnvelopeWorkspace(raw).result(True, True, False)
    with np.errstate(over="raise", under="ignore", invalid="raise", divide="raise"):
        actual = EnvelopeWorkspace(raw*scale).result(True, True, False)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_an_exact_zero_sample_has_explicit_log_and_reference_behavior():
    # Analytic input avoids treating Hilbert roundoff as an exact-zero fixture.
    z = np.array([[0, 1+1j, 2+1j, 1+2j], [1j, 2+2j, 3+1j, 2+3j]])
    work = EnvelopeWorkspace(z)
    assert np.isfinite(work.result(False, False, False)[0, 1])
    assert np.isnan(work.result(False, True, False)[0, 1])
    assert work.diagnostics(False, True)["reason"][0, 1] == "zero_log_envelope"
    assert np.isnan(work.result(True, False, False)[0, 1])
    assert work.diagnostics(True, False)["directional_reason"][0, 1] == "zero_reference_envelope"


def test_shared_hilbert_and_directional_work_and_cache_ownership():
    raw = np.random.default_rng(5).normal(size=(4, 256))
    data = Data(raw, normalise=False)
    with patch.object(implementation, "hilbert", wraps=implementation.hilbert) as analytic:
        with patch.object(EnvelopeWorkspace, "_orthogonal_directions", autospec=True,
                          side_effect=EnvelopeWorkspace._orthogonal_directions) as directions:
            for orth, log, absolute in SPECS:
                spi = PowerEnvelopeCorrelation(orth=orth, log=log, absolute=absolute)
                expected = spi.multivariate(data)
                returned = spi.multivariate(data)
                returned[:] = 123
                diagnostic = spi.diagnostics(data)
                diagnostic["reason"][:] = "changed"
                np.testing.assert_allclose(spi.multivariate(data), expected, atol=0)
                assert spi.diagnostics(data)["reason"][0, 1] == "estimated"
    assert analytic.call_count == directions.call_count == 1


@pytest.mark.parametrize("mutation", ["view", "replace"])
def test_mutation_invalidates_envelope_cache(mutation):
    raw = np.random.default_rng(6).normal(size=(3, 128))
    data = Data(raw, normalise=False)
    spi = PowerEnvelopeCorrelation(orth=True, log=True)
    spi.multivariate(data)
    old = envelope_workspace(data)
    if mutation == "view":
        data.to_numpy()[0, :, 0] += np.random.default_rng(7).normal(size=128)
    else:
        data.set_data(np.random.default_rng(7).normal(size=(3, 128)))
    expected = spi.multivariate(Data(data.to_numpy().copy(), dim_order="psr", normalise=False))
    np.testing.assert_allclose(spi.multivariate(data), expected, atol=1e-13)
    assert envelope_workspace(data) is not old


def test_channel_permutation_and_one_pair_blocks():
    raw = np.random.default_rng(8).normal(size=(5, 129))
    expected = EnvelopeWorkspace(raw)
    blocked = EnvelopeWorkspace(raw)
    blocked.block_bytes = 129*8
    order = [3, 1, 4, 0, 2]
    permuted = EnvelopeWorkspace(raw[order])
    for orth, log, absolute in SPECS:
        result = expected.result(orth, log, absolute)
        np.testing.assert_allclose(blocked.result(orth, log, absolute), result, atol=1e-13)
        np.testing.assert_allclose(permuted.result(orth, log, absolute), result[np.ix_(order, order)], atol=1e-13)


def test_nonfinite_channels_do_not_poison_healthy_pairs():
    raw = np.random.default_rng(9).normal(size=(3, 128))
    raw[2, 20] = np.nan
    work = EnvelopeWorkspace(raw)
    for orth, log, absolute in SPECS:
        result = work.result(orth, log, absolute)
        assert np.isfinite(result[0, 1]) and np.isnan(result[0, 2])
        assert work.diagnostics(orth, log)["reason"][0, 2] == "nonfinite_input"


def test_instance_labels_are_signed_unless_absolute_and_do_not_leak():
    for orth, log, absolute in reversed(SPECS):
        spi = PowerEnvelopeCorrelation(orth=orth, log=log, absolute=absolute)
        assert spi.issigned() is (not absolute)
        assert ("signed" in spi.labels) is (not absolute)
        assert ("unsigned" in spi.labels) is absolute
    assert PowerEnvelopeCorrelation.labels == ["misc", "undirected"]
    with pytest.raises(ValueError, match="orthogonalization"):
        PowerEnvelopeCorrelation(absolute=True)
