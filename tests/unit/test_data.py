"""Unit tests for pyspi.data: preprocessing order, validation and loaders.

These lock current behaviour (including quirks) of ``pyspi.data.Data``.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import detrend
from scipy.stats import zscore

from pyspi.data import Data, load_dataset

NONFINITE_MSG = "non-finite values (NaNs or infinity)"


def _random(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def test_dim_order_sp_transposes():
    x = _random((50, 3), seed=1)  # observations x processes
    d = Data(x, dim_order="sp", normalise=False)
    assert (d.n_processes, d.n_observations, d.n_replications) == (3, 50, 1)
    np.testing.assert_array_equal(d.to_numpy(squeeze=True), x.T)


def test_truncation_happens_before_normalisation():
    x = _random((5, 100), seed=2) * 3.0 + 7.0
    d = Data(x, n_processes=3, n_observations=60, normalise=True)
    expected = zscore(x[:3, :60], axis=1, ddof=1)
    out = d.to_numpy(squeeze=True)
    assert out.shape == (3, 60)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    # Normalising the full rows first and truncating afterwards gives different numbers.
    assert not np.allclose(out, zscore(x, axis=1, ddof=1)[:3, :60])


def test_detrend_then_zscore_order():
    # Strong per-row linear trends and very different per-row scales, so that
    # detrend(zscore(x)) != zscore(detrend(x)) by a clear margin.
    rng = np.random.default_rng(3)
    t = np.arange(80, dtype=float)
    slopes = np.array([0.5, -2.0, 5.0, 0.1])[:, None]
    scales = np.array([1.0, 10.0, 0.1, 100.0])[:, None]
    x = slopes * t + scales * rng.standard_normal((4, 80)) + 3.0

    out = Data(x, detrend=True, normalise=True).to_numpy(squeeze=True)

    expected = zscore(detrend(x, axis=1), axis=1, ddof=1)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    reverse = detrend(zscore(x, axis=1, ddof=1), axis=1)
    assert not np.allclose(out, reverse, rtol=1e-3, atol=1e-3)


def test_nan_raises_value_error():
    x = _random((3, 50), seed=4)
    x[1, 10] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        Data(x)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inf_raises_value_error():
    # Reject infinity before preprocessing.
    x = _random((3, 50), seed=5)
    x[2, 7] = np.inf
    with pytest.raises(ValueError) as excinfo:
        Data(x)
    assert NONFINITE_MSG in str(excinfo.value)
    assert "[2]" in str(excinfo.value)


@pytest.mark.parametrize("normalise,detrended", [(False, False), (True, False), (False, True), (True, True)])
def test_add_process_preprocesses_every_channel(normalise, detrended):
    rng = np.random.default_rng(6)
    first = rng.standard_normal(60) * 4.0 + 2.0
    second = rng.standard_normal(60) * 4.0 + 2.0

    d = Data(normalise=normalise, detrend=detrended)
    d.add_process(first)
    before = d.to_numpy().copy()
    d.add_process(second)

    assert (d.n_processes, d.n_observations, d.n_replications) == (2, 60, 1)
    arr = d.to_numpy(squeeze=True)
    expected = np.stack([first, second])
    if detrended:
        expected = detrend(expected, axis=1)
    if normalise:
        expected = zscore(expected, axis=1, ddof=1)
    np.testing.assert_allclose(arr, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(d.to_numpy()[:1], before)


def test_add_process_rejects_2d():
    d = Data(_random((2, 50), seed=7))
    with pytest.raises(TypeError, match="1D numpy array"):
        d.add_process(_random((2, 50), seed=8))


@pytest.mark.parametrize("kind", ["ndarray", "dataframe", "npy", "txt"])
def test_convert_to_numpy(kind, tmp_path):
    x = _random((3, 20), seed=9)
    if kind == "ndarray":
        src = x
    elif kind == "dataframe":
        src = pd.DataFrame(x)
    elif kind == "npy":
        src = str(tmp_path / "x.npy")
        np.save(src, x)
    else:
        src = str(tmp_path / "x.txt")
        np.savetxt(src, x, fmt="%.18e")

    out = Data.convert_to_numpy(src)
    assert isinstance(out, np.ndarray)
    if kind == "ndarray":
        assert out is x
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize("name, shape", [("forex", (7, 250)), ("cml", (10, 500))])
def test_load_dataset_shapes(name, shape):
    d = load_dataset(name)
    assert (d.n_processes, d.n_observations, d.n_replications) == shape + (1,)


def test_load_dataset_unknown_name_raises():
    with pytest.raises(NameError, match="Unknown dataset"):
        load_dataset("no_such_dataset")


def test_to_numpy_squeeze_shapes():
    d = Data(_random((4, 30), seed=10))
    assert d.to_numpy().shape == (4, 30, 1)
    assert d.to_numpy(squeeze=True).shape == (4, 30)
    assert d.to_numpy(realisation=0).shape == (4, 30)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("normalise,detrended", [(False, False), (True, False), (False, True), (True, True)])
def test_nonfinite_input_rejected_under_all_flags(value, normalise, detrended):
    x = _random((3, 1024))
    x[2, 11] = value
    with pytest.raises(ValueError, match=r"non-finite values.*processes: \[2\]"):
        Data(x, normalise=normalise, detrend=detrended)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["zero", "constant", "small_constant", "linear"])
def test_zscore_rejects_unusable_variance(dtype, kind):
    t = np.arange(1024, dtype=dtype)
    x = _random((3, 1024)).astype(dtype)
    x[1] = {"zero": np.zeros_like(t), "constant": np.full_like(t, 3.),
            "small_constant": np.full_like(t, .001), "linear": .37*t + 1.1}[kind]
    with pytest.raises(ValueError, match=r"zero or numerically unusable variance.*processes: \[1\]"):
        Data(x, detrend=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1., 1e6])
def test_zscore_preserves_valid_signals_and_sample_sd(dtype, scale):
    x = (_random((3, 1024)) * scale).astype(dtype)
    np.testing.assert_array_equal(
        Data(x).to_numpy(squeeze=True), zscore(x, axis=1, ddof=1)
    )


@pytest.mark.parametrize("value", [0., 3., .001])
def test_constant_channel_rejected_without_detrend(value):
    x = _random((3, 1024))
    x[1] = value
    with pytest.raises(ValueError, match="zero or numerically unusable variance"):
        Data(x)


def test_zscore_requires_two_samples():
    with pytest.raises(ValueError, match="at least two observations"):
        Data(np.ones((3, 1)))


def test_invalid_append_leaves_existing_data_unchanged():
    data = Data(_random((2, 1024)))
    before = data.to_numpy().copy()
    bad = _random((1024,))
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        data.add_process(bad)
    np.testing.assert_array_equal(data.to_numpy(), before)
