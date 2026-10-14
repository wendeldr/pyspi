"""Production API invariants independent of comparison helpers and metric math."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyspi import storage
from pyspi.analysis import time_mean

from storage_cases import assert_bits, matrix_case, special_values, tree_fingerprint


def test_public_api_has_no_analysis_or_lossy_conversion():
    assert not hasattr(storage.Results, "mean")
    assert not hasattr(storage, "time_mean")
    assert not hasattr(storage, "pack")
    assert not hasattr(storage, "float32_range_problem")


def test_store_diagnostic_name_is_reserved(tmp_path):
    path = tmp_path / "reserved"
    with pytest.raises(ValueError):
        storage.write_metric(path, "__store__", np.eye(2))
    assert not path.exists()


def test_mixed_precision_metadata_and_order_are_preserved(tmp_path):
    a = matrix_case("full", dtype=np.float32)
    b = matrix_case("anti", dtype=np.float64)
    recording = {"channels": ["B", "A", "Z", "C"], "custom": {"order": [3, 1, 2, 0]}}
    attrs = {"z": {"n": "user field", "mode": "scientific description", "tags": ["b", "a"]}}
    storage.write_results(tmp_path, {"z": a, "a": b}, attrs=attrs, recording=recording)
    attrs["z"]["tags"].append("later")
    recording["channels"].reverse()
    reader = storage.open_results(tmp_path)
    assert reader.metrics == ["z", "a"]
    assert_bits(reader.load("z"), a)
    assert_bits(reader.load("a"), b)
    assert reader.info("z")["attrs"] == {"n": "user field", "mode": "scientific description", "tags": ["b", "a"]}
    assert reader.recording["channels"] == ["B", "A", "Z", "C"]
    storage.write_metric(tmp_path, "third", b)
    storage.write_metric(tmp_path, "z", a, overwrite=True)
    assert reader.metrics == ["z", "a", "third"]


@pytest.mark.parametrize("bad", [
    {"attrs": {"z": {"bad": np.nan}}}, {"recording": {"bad": np.inf}},
    {"attrs": {"z": {1: "nonstring"}}}, {"labels": {"z": "one string"}},
    {"codec": "unsupported"}, {"level": True}, {"level": 13},
    {"attrs": {"absent": {}}},
])
def test_invalid_batch_options_preserve_existing_store(tmp_path, bad):
    data = matrix_case("sym")
    storage.write_metric(tmp_path, "z", data)
    before = tree_fingerprint(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        storage.write_results(tmp_path, {"z": data, "a": data}, overwrite=True, **bad)
    assert tree_fingerprint(tmp_path) == before
    assert storage.verify(tmp_path) == {}


def calculator_table(values, processes=("B", "A"), metric_order=("z", "a")):
    columns = pd.MultiIndex.from_product([metric_order, processes], names=["spi", "process"])
    table = pd.DataFrame(np.concatenate([values[name] for name in metric_order], axis=1),
                         index=list(processes), columns=columns)
    labels = {name: ["test", name] for name in metric_order}
    return SimpleNamespace(table=table, get_stat_labels=lambda: labels)


def test_calculator_adapter_aligns_axes_and_preserves_requested_order(tmp_path):
    first = {"z": np.array([[np.nan, -2.0], [7.0, np.inf]]),
             "a": np.array([[1.0, 2.0], [3.0, -0.0]])}
    second = {name: values + 10 for name, values in first.items()}
    c1 = calculator_table(first)
    c2 = calculator_table({name: data[::-1, ::-1] for name, data in second.items()},
                          processes=("A", "B"), metric_order=("a", "z"))
    before = c1.table.copy(deep=True)
    storage.write_calculator(tmp_path, [c1, c2], metrics=["a", "z"],
                             channel_names=["electrode B", "electrode A"],
                             recording={"window_onsets": [1024, 2048], "sampling_rate": 2048},
                             attrs={"a": {"parameters": {"lag": 3}}})
    reader = storage.open_results(tmp_path)
    assert reader.metrics == ["a", "z"]
    assert reader.recording["channels"] == ["electrode B", "electrode A"]
    assert reader.recording["window_onsets"] == [1024, 2048]
    for name in first:
        assert_bits(reader.load(name), np.stack([first[name], second[name]], axis=2))
        assert reader.info(name)["labels"] == ["test", name]
    pd.testing.assert_frame_equal(c1.table, before)


def test_single_calculator_and_one_window_sequence_have_distinct_shapes(tmp_path):
    values = {"z": np.eye(2), "a": -np.eye(2)}
    calc = calculator_table(values)
    storage.write_calculator(tmp_path / "single", calc)
    storage.write_calculator(tmp_path / "windows", [calc])
    assert_bits(storage.open_results(tmp_path / "single").load("z"), values["z"])
    assert_bits(storage.open_results(tmp_path / "windows").load("z"), values["z"][..., None])


@pytest.mark.parametrize("problem", ["empty", "processes", "labels", "metric", "names", "metadata", "duplicates", "columns", "dtype",
                                     "duplicate_process", "empty_process", "flat_columns", "string_metrics", "wrong_columns", "integer_values"])
def test_adapter_rejects_ambiguous_tables_before_creating_store(tmp_path, problem):
    values = {"z": np.eye(2), "a": -np.eye(2)}
    c1, c2 = calculator_table(values), calculator_table(values)
    calculators, kwargs = [c1, c2], {}
    if problem == "empty":
        calculators = []
    elif problem == "processes":
        c2.table.index = ["B", "C"]
    elif problem == "labels":
        c2.get_stat_labels = lambda: {"z": ["different"], "a": ["test", "a"]}
    elif problem == "metric":
        kwargs["metrics"] = ["absent"]
    elif problem == "names":
        kwargs["channel_names"] = ["one"]
    elif problem == "metadata":
        kwargs["recording"] = {"channels": ["wrong", "order"]}
    elif problem == "duplicates":
        kwargs["metrics"] = ["z", "z"]
    elif problem == "columns":
        c2.table.columns = pd.MultiIndex.from_tuples([("z", "B")] * 4)
    elif problem == "dtype":
        c2.table = c2.table.astype(np.float32)
    elif problem == "duplicate_process":
        c1.table.index = ["B", "B"]
    elif problem == "empty_process":
        c1.table = c1.table.iloc[:0]
    elif problem == "flat_columns":
        c1.table.columns = ["one", "two", "three", "four"]
    elif problem == "string_metrics":
        kwargs["metrics"] = "z"
    elif problem == "wrong_columns":
        c2.table.columns = pd.MultiIndex.from_product([["z", "a"], ["B", "C"]])
    elif problem == "integer_values":
        c2.table = c2.table.astype(np.int64)
    path = tmp_path / "bad"
    with pytest.raises((ValueError, KeyError, TypeError)):
        storage.write_calculator(path, calculators, **kwargs)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_analysis_time_mean_matches_float64_numpy(store, force_mode, tmp_path, kind, dtype):
    force_mode(kind)
    data = matrix_case(kind, windows=17, dtype=dtype)
    data[0, 0, 0] = np.nan
    storage.write_metric(tmp_path, "metric", data)
    expected = np.nanmean(data, axis=2, dtype=np.float64)
    assert_bits(time_mean(storage.open_results(tmp_path), "metric"), expected)


def test_analysis_antisymmetric_subnormal_mean_preserves_negative_zero(tmp_path):
    data = np.zeros((2, 2, 3))
    data[0, 1, 0] = np.nextafter(0.0, 1.0)
    data[1, 0, 0] = -data[0, 1, 0]
    storage.write_metric(tmp_path, "anti", data)
    reader = storage.open_results(tmp_path)
    assert reader.info("anti")["mode"] == "anti"
    actual = time_mean(reader, "anti")
    assert_bits(actual, np.nanmean(data, axis=2))
    assert np.signbit(actual[1, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_analysis_special_values_match_numpy(tmp_path, dtype):
    data = np.resize(special_values(dtype), (4, 4, 33)).copy()
    storage.write_metric(tmp_path, "special", data)
    with np.errstate(invalid="ignore"):
        expected = np.nanmean(data, axis=2, dtype=np.float64)
        assert_bits(time_mean(storage.open_results(tmp_path), "special"), expected)


@pytest.mark.parametrize("windows", [0, 1, 9])
def test_analysis_invalid_series_and_single_matrices(tmp_path, windows):
    data = np.full((2, 2, windows), np.nan)
    storage.write_metric(tmp_path, "nan", data)
    actual = time_mean(storage.open_results(tmp_path), "nan")
    assert actual.shape == (2, 2) and np.isnan(actual).all()
    storage.write_metric(tmp_path, "single", np.eye(2, dtype=np.float32))
    assert_bits(time_mean(storage.open_results(tmp_path), "single"), np.eye(2))


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_analysis_reduction_crosses_column_blocks_and_remainders(store, force_mode, monkeypatch, tmp_path, kind, dtype):
    from pyspi import analysis
    monkeypatch.setattr(analysis, "MEAN_BLOCK_COLUMNS", 3)
    force_mode(kind)
    data = matrix_case(kind, n=5, windows=13, dtype=dtype)
    data[0, 0, 0] = np.nan
    storage.write_metric(tmp_path, "blocks", data)
    expected = np.nanmean(data, axis=2, dtype=np.float64)
    assert_bits(time_mean(storage.open_results(tmp_path), "blocks"), expected)
