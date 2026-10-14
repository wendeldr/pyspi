"""Exercise calculation and exact storage in the unified environment."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.java
def test_calculated_windows_roundtrip_exactly(tmp_path, monkeypatch, seed_all):
    for name in ("zarr", "numcodecs", "pcodec", "zstandard", "blosc2", "zfpy", "fpzip"):
        pytest.importorskip(name)
    from pyspi.calculator import Calculator

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "storage_comparison"))
    store = importlib.import_module("pspi_store")
    raw = np.random.default_rng(112).normal(size=(4, 2048))
    windows = {}
    for start in (0, 1024):
        calc = Calculator(dataset=raw[:, start:start + 1024], subset="fabfour", detrend=True, normalise=True)
        seed_all(42)
        calc.compute()
        assert len(calc.spis) == 4
        for name in calc.spis:
            values = calc.table[name].to_numpy()
            assert np.isfinite(values[~np.eye(4, dtype=bool)]).all(), name
            windows.setdefault(name, []).append(values.copy())

    results = {name: np.stack(values, axis=-1) for name, values in windows.items()}
    path = tmp_path / "calculated.zarr"
    store.write_results(path, results, codec="pcodec")
    reader = store.Results(path)
    assert store.verify(path) == {}
    for name, expected in results.items():
        actual, available = reader.load(name, return_availability=True)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert available.all()  # Calculated NaNs on the diagonal are present.
        subset = reader.load(name, [1, 0, 1])
        np.testing.assert_array_equal(subset.view(np.uint64), expected[..., [1, 0, 1]].view(np.uint64))
