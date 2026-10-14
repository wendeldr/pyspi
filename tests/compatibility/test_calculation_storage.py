"""Exercise calculation and exact storage in the unified environment."""

import numpy as np
import pytest


@pytest.mark.java
def test_calculated_windows_roundtrip_exactly(tmp_path, seed_all):
    pytest.importorskip("zarr")
    from pyspi.calculator import Calculator
    from pyspi import storage as store
    raw = np.random.default_rng(112).normal(size=(4, 2048))
    windows = {}
    calculators = []
    for start in (0, 1024):
        calc = Calculator(dataset=raw[:, start:start + 1024], subset="fabfour", detrend=True, normalise=True)
        seed_all(42)
        calc.compute()
        calculators.append(calc)
        assert len(calc.spis) == 4
        for name in calc.spis:
            values = calc.table[name].to_numpy()
            assert np.isfinite(values[~np.eye(4, dtype=bool)]).all(), name
            windows.setdefault(name, []).append(values.copy())

    results = {name: np.stack(values, axis=-1) for name, values in windows.items()}
    path = tmp_path / "calculated.zarr"
    store.write_calculator(path, calculators, codec="pcodec",
                           channel_names=["A", "B", "C", "D"],
                           recording={"sampling_rate": 2048, "window_onsets": [0, 1024]})
    reader = store.Results(path)
    assert store.verify(path) == {}
    assert reader.metrics == list(results)
    assert reader.recording == {"channels": ["A", "B", "C", "D"],
                                "sampling_rate": 2048, "window_onsets": [0, 1024]}
    for name, expected in results.items():
        actual, available = reader.load(name, return_availability=True)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert available.all()  # Calculated NaNs on the diagonal are present.
        subset = reader.load(name, [1, 0, 1])
        np.testing.assert_array_equal(subset.view(np.uint64), expected[..., [1, 0, 1]].view(np.uint64))
        assert reader.info(name)["labels"] == calculators[0].get_stat_labels()[name]

    single = tmp_path / "single.zarr"
    store.write_calculator(single, calculators[0], metrics=list(results)[::-1])
    assert store.open_results(single).metrics == list(results)[::-1]
    for name, expected in results.items():
        np.testing.assert_array_equal(store.open_results(single).load(name).view(np.uint64),
                                      expected[..., 0].view(np.uint64))
