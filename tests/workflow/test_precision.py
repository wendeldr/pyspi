"""Output precision is explicit and independent of calculation precision."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pyspi.workflow._config import WorkflowError, load_config
from pyspi.workflow._precision import convert_output, resolve_dtypes
from pyspi.workflow._preflight import preflight
from pyspi.workflow._run import _batch_size, _store_values


def test_required_dtype_and_override_validation(edf, workflow):
    path = workflow(edf(), output_dtype="float32")
    cfg = load_config(path)
    report = preflight(cfg)
    identifier = report.metrics[0].identifier
    selected = replace(cfg, output_dtype_overrides={identifier: "float64"})
    assert resolve_dtypes(selected, [identifier]) == {identifier: np.dtype("float64").str}
    with pytest.raises(WorkflowError, match="unselected metrics"):
        preflight(replace(cfg, output_dtype_overrides={"misspelled": "float64"}))
    data = yaml.safe_load(path.read_text())
    del data["output_dtype"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(WorkflowError, match="output_dtype.*required"):
        load_config(path)


@pytest.mark.parametrize("dtype", ["f32", "float128", None, 32, True])
def test_invalid_dtype_rejected(dtype, edf, workflow):
    with pytest.raises(WorkflowError, match="output_dtype"):
        load_config(workflow(edf(), output_dtype=dtype))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_conversion_and_mask_preserve_source(dtype):
    values = np.array([[np.nan, -0.123456789], [np.inf, -0.0]])
    original = values.copy()
    buffers = {}
    _store_values(buffers, SimpleNamespace(identifier="metric"), values,
                  np.array([2, 0]), 3, 2, 1, {"metric": dtype})
    result = buffers["metric"]
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result[np.ix_([2, 0], [2, 0], [1])][:, :, 0], values.astype(dtype))
    assert np.isnan(result[1]).all() and np.isnan(result[:, :, 0]).all()
    np.testing.assert_array_equal(values.view(np.uint64), original.view(np.uint64))


def test_overflow_is_logged_and_nonfinite_inputs_are_not_overflows():
    x, event = convert_output(np.array([1e308, -1e308, np.nan, np.inf, -np.inf, 1e-100]),
                              "float32", "metric")
    assert np.isposinf(x[0]) and np.isneginf(x[1])
    assert np.isnan(x[2]) and np.isposinf(x[3]) and np.isneginf(x[4]) and x[5] == 0
    assert event["positions"] == [[0], [1]]
    assert event["original_values"] == [1e308, -1e308]
    assert event["count"] == 2 and event["positive_count"] == event["negative_count"] == 1
    _, clean = convert_output(np.array([np.nan, np.inf, -np.inf]), "float32", "metric")
    assert clean is None


def test_overflow_positions_follow_saved_channel_order():
    buffers = {}
    event = _store_values(buffers, SimpleNamespace(identifier="metric"),
                          np.array([[np.nan, 1e308], [-1e308, np.nan]]),
                          np.array([2, 0]), 3, 1, 0, {"metric": "float32"})
    assert event["positions"] == [[2, 0], [0, 2]]
    assert event["original_values"] == [1e308, -1e308]
    assert event["position_axes"] == "zero_based_output_row_column"
    assert np.isposinf(buffers["metric"][2, 0, 0])
    assert np.isneginf(buffers["metric"][0, 2, 0])


def test_mixed_precision_batch_budget(edf, workflow):
    cfg = load_config(workflow(edf(), output_dtype="float16", output_batch_mib=1,
                               output_dtype_overrides={"double": "float64"}))
    size = _batch_size(cfg, 118, ["half", "double"])
    assert size == (1024 ** 2) // (118 * 118 * (2 + 8))
