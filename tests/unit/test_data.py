"""Unit tests for pyspi.data: preprocessing order, validation and loaders.

These lock current behaviour (including quirks) of ``pyspi.data.Data``.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import detrend
from scipy.stats import zscore

from pyspi.data import Data, load_dataset

NAN_MSG = "non-numerics (NaNs) in processes"


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
    with pytest.raises(ValueError, match=r"non-numerics \(NaNs\) in processes"):
        Data(x)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inf_raises_value_error():
    # zscore turns a row containing inf into NaN, which then trips the NaN check.
    x = _random((3, 50), seed=5)
    x[2, 7] = np.inf
    with pytest.raises(ValueError) as excinfo:
        Data(x)
    assert NAN_MSG in str(excinfo.value)
    assert "[2]" in str(excinfo.value)


def test_add_process_appends_raw_unnormalised():
    rng = np.random.default_rng(6)
    first = rng.standard_normal(60) * 4.0 + 2.0
    second = rng.standard_normal(60) * 4.0 + 2.0

    d = Data()
    d.add_process(first)
    d.add_process(second)

    assert (d.n_processes, d.n_observations, d.n_replications) == (2, 60, 1)
    arr = d.to_numpy(squeeze=True)
    # First process goes through set_data (normalise=True by default) ...
    np.testing.assert_allclose(arr[0], zscore(first, ddof=1), rtol=1e-12, atol=1e-12)
    # ... but later processes are appended raw.
    np.testing.assert_array_equal(arr[1], second)


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
