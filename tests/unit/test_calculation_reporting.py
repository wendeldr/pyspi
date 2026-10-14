"""Completion diagnostics and the boundary between calculation and consumers.

Synthetic SPIs control success/failure independently of their numeric values.
Existing golden tests additionally exercise real estimators and shared caches.
"""

from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import warnings
import weakref

import numpy as np
import pandas as pd
import pytest
import yaml

import pyspi.calculator as calculator_module
from pyspi.calculator import Calculator
from pyspi.calculation import MetricReport
from pyspi.data import Data


class Progress:
    def __init__(self, identifiers):
        self.identifiers = identifiers
        self.closed = False
        self.format_dict = {"elapsed": 0.0}

    def __iter__(self):
        return iter(self.identifiers)

    def set_description(self, description):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def make_calculator(tmp_path, monkeypatch):
    path = tmp_path / "one.yaml"
    path.write_text(yaml.safe_dump({".statistics.basic": {
        "Covariance": {"configs": [{"estimator": "EmpiricalCovariance"}]}}}))
    monkeypatch.setattr(calculator_module, "tqdm", Progress)

    def make(functions, *, processes=3):
        calc = Calculator(configfile=str(path), normalise=False)
        calc._spis = {name: SimpleNamespace(multivariate=fn)
                      for name, fn in functions.items()}
        data = Data(np.random.default_rng(131).normal(size=(processes, 32)), normalise=False)
        data._procnames = [f"channel-{processes-i}" for i in range(processes)]
        calc.load_dataset(data)
        return calc

    return make


def matrix(data):
    return np.arange(data.n_processes ** 2, dtype=float).reshape(data.n_processes, -1)


def fail(data):
    warnings.warn("before failure", RuntimeWarning)
    try:
        raise KeyError("inner cause")
    except KeyError as cause:
        raise ValueError("estimator failed") from cause


def test_success_nan_and_failure_are_distinct_finished_attempts(make_calculator):
    events = []
    calls = []

    def record(name, fn):
        def call(data):
            calls.append(name)
            return fn(data)
        return call

    functions = {"numeric": matrix, "nan": lambda d: np.full((3, 3), np.nan),
                 "failed": fail, "after": matrix}
    calc = make_calculator({name: record(name, fn) for name, fn in functions.items()})
    assert calc.metric_reports == {}

    def consume(report, values):
        assert isinstance(report, MetricReport)
        assert calc.metric_reports[report.identifier] is report
        assert list(calc.metric_reports) == calls  # Published before callback.
        events.append((report, values))

    with pytest.warns((UserWarning, RuntimeWarning)) as visible:
        result = calc.compute(on_metric_complete=consume)
    assert result is None
    assert calls == list(functions) == list(calc.metric_reports)
    assert [report.identifier for report, _ in events] == list(functions)
    assert [report.status for report, _ in events] == ["success", "success", "failed", "success"]
    assert np.isnan(events[1][1]).all()
    assert events[2][1] is None
    assert np.isnan(calc.table["failed"].to_numpy()).all()
    for report, values in events:
        assert np.isfinite(report.elapsed_seconds) and report.elapsed_seconds >= 0
        if report.status == "success":
            assert report.error is None
            np.testing.assert_array_equal(values, calc.table[report.identifier].to_numpy())
            assert np.isnan(np.diag(values)).all()
            assert not values.flags.writeable
    error = events[2][0].error
    assert error.type == "builtins.ValueError"
    assert error.message == "estimator failed"
    assert "KeyError: 'inner cause'" in error.traceback
    assert "ValueError: estimator failed" in error.traceback
    assert "test_calculation_reporting.py" in error.traceback
    assert [str(w.message) for w in visible] == [
        "before failure", 'Caught <class \'ValueError\'> for SPI "failed": estimator failed']
    assert [w.message for w in events[2][0].warnings] == ["before failure"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_callback_snapshot_preserves_bits_and_is_detached(make_calculator, dtype):
    uint = np.uint32 if dtype == np.float32 else np.uint64
    nan_word = 0x7FC01234 if dtype == np.float32 else 0x7FF8000000001234
    source = np.array([[1.0, -0.0, np.inf], [-np.inf, 2.0, 0.0],
                       [3.0, 4.0, 5.0]], dtype=dtype)
    source.view(uint)[2, 1] = nan_word
    expected = source.copy()
    np.fill_diagonal(expected, np.nan)
    snapshots = []
    calc = make_calculator({"metric": lambda d: source})

    def consume(report, values):
        assert values.dtype == expected.dtype
        assert not np.shares_memory(values, source)
        assert not np.shares_memory(values, calc.table["metric"].to_numpy())
        np.testing.assert_array_equal(values.view(uint), expected.view(uint))
        snapshots.append(values)
        # A consumer can own/re-enable its copy; it cannot alter Calculator.
        values.flags.writeable = True
        values[0, 1] = 900
        np.testing.assert_array_equal(calc.table["metric"].to_numpy().view(uint), expected.view(uint))

    calc.compute(on_metric_complete=consume)
    source[:] = 123
    calc.load_dataset(np.zeros((3, 32)))
    assert snapshots[0][0, 1] == 900
    assert np.isposinf(snapshots[0][0, 2])
    assert snapshots[0].view(uint)[2, 1] == nan_word


def test_reports_are_small_immutable_and_serializable(make_calculator):
    calc = make_calculator({"failed": fail})
    with pytest.warns((UserWarning, RuntimeWarning)):
        calc.compute()
    reports = calc.metric_reports
    report = reports["failed"]
    with pytest.raises(TypeError):
        reports["extra"] = report
    with pytest.raises(FrozenInstanceError):
        report.status = "success"
    with pytest.raises(FrozenInstanceError):
        report.error.message = "changed"
    with pytest.raises(FrozenInstanceError):
        report.warnings[0].message = "changed"
    diagnostic = json.loads(json.dumps(report.as_dict(), allow_nan=False))
    assert set(diagnostic) == {"identifier", "status", "elapsed_seconds", "warnings", "error"}
    assert diagnostic["identifier"] == "failed"
    assert diagnostic["warnings"][0]["category"] == "builtins.RuntimeWarning"
    assert diagnostic["warnings"][0]["filename"] == __file__
    assert isinstance(diagnostic["warnings"][0]["lineno"], int)
    assert diagnostic["error"]["type"] == "builtins.ValueError"
    diagnostic["warnings"][0]["message"] = "changed"
    assert report.warnings[0].message == "before failure"


def test_reports_reset_on_new_compute_and_dataset(make_calculator):
    calc = make_calculator({"one": matrix, "two": matrix})
    reports = calc.metric_reports
    calc.compute()
    old_report = reports["one"]
    assert list(reports) == ["one", "two"]

    def stop_after_first(report, values):
        raise OSError("consumer stopped")

    with pytest.raises(OSError, match="consumer stopped"):
        calc.compute(on_metric_complete=stop_after_first)
    assert list(reports) == ["one"]
    assert reports["one"] is not old_report
    calc.load_dataset(np.zeros((3, 32)))
    assert reports == {} == calc.metric_reports
    assert old_report.status == "success"


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("exception", [OSError("disk full"), AttributeError("consumer bug"), KeyboardInterrupt("stop")])
def test_callback_errors_propagate_without_becoming_metric_failures(
        make_calculator, monkeypatch, failed, exception):
    started = []

    def first(data):
        started.append("first")
        if failed:
            raise RuntimeError("metric failure")
        return matrix(data)

    def second(data):
        pytest.fail("computation continued after consumer failed")

    bars = []
    def progress(keys):
        bars.append(Progress(keys))
        return bars[-1]
    monkeypatch.setattr(calculator_module, "tqdm", progress)
    calc = make_calculator({"first": first, "second": second})
    showwarning = warnings.showwarning

    def consume(report, values):
        assert report.status == ("failed" if failed else "success")
        if failed:
            assert report.error.message == "metric failure" and values is None
        raise exception

    with pytest.raises(type(exception)) as raised:
        calc.compute(on_metric_complete=consume)
    assert raised.value is exception
    assert started == ["first"]
    assert list(calc.metric_reports) == ["first"]
    assert calc.metric_reports["first"].status == ("failed" if failed else "success")
    assert bars[0].closed
    assert warnings.showwarning is showwarning


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_metric_has_no_finished_report(make_calculator, monkeypatch, interruption):
    def stop(data):
        raise interruption("stop")
    bars = []
    def progress(keys):
        bars.append(Progress(keys))
        return bars[-1]
    monkeypatch.setattr(calculator_module, "tqdm", progress)
    calc = make_calculator({"complete": matrix, "interrupted": stop, "unattempted": matrix})
    events = []
    showwarning = warnings.showwarning
    with pytest.raises(interruption):
        calc.compute(on_metric_complete=lambda r, v: events.append(r.identifier))
    assert list(calc.metric_reports) == events == ["complete"]
    assert warnings.showwarning is showwarning
    assert bars[0].closed


def test_metric_timing_excludes_consumer_time(make_calculator, monkeypatch):
    clock = [20.0]
    monkeypatch.setattr(calculator_module, "time", SimpleNamespace(
        perf_counter=lambda: clock[0], time=lambda: -999.0))
    def calculate(data):
        clock[0] += 2.25
        return matrix(data)
    def consume(report, values):
        clock[0] += 100
    original = pd.DataFrame.to_numpy
    def copy(self, *args, **kwargs):
        if kwargs.get("copy", False):
            clock[0] += 40
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_numpy", copy)
    calc = make_calculator({"one": calculate, "two": calculate})
    # Wall-clock timestamps are unnecessary and must not determine durations.
    calc.compute(on_metric_complete=consume)
    assert [r.elapsed_seconds for r in calc.metric_reports.values()] == [2.25, 2.25]


def test_warning_filters_visibility_registry_and_handler_are_preserved(make_calculator):
    # One source location used by two metrics and another compute call. A
    # per-metric catch_warnings() would invalidate registries and emit it again.
    scope = {"__name__": "reporting_warning_test", "warnings": warnings, "np": np}
    exec(compile('def calculate(data):\n    warnings.warn("once", RuntimeWarning)\n'
                 '    return np.ones((3, 3))\n', "<reporting_warning_test>", "exec"), scope)
    calc = make_calculator({"one": scope["calculate"], "two": scope["calculate"]})
    with warnings.catch_warnings(record=True) as visible:
        warnings.simplefilter("default")
        handler = warnings.showwarning
        filters = warnings.filters.copy()
        calc.compute()
        assert [len(r.warnings) for r in calc.metric_reports.values()] == [1, 0]
        calc.compute()
        # Existing pandas operations between compute calls invalidate warning
        # registries; the pre-instrumentation Calculator also emits once/call.
        assert [len(r.warnings) for r in calc.metric_reports.values()] == [1, 0]
        assert warnings.filters == filters
        assert warnings.showwarning is handler
    assert [str(w.message) for w in visible] == ["once", "once"]


def test_ignored_warning_is_not_reported_and_custom_handler_is_forwarded(make_calculator, monkeypatch):
    def calculate(data):
        warnings.warn("hidden", DeprecationWarning)
        warnings.warn("visible", RuntimeWarning)
        return matrix(data)
    visible = []
    def handler(message, category, filename, lineno, file=None, line=None):
        visible.append((str(message), category, filename, lineno))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("always", RuntimeWarning)
        monkeypatch.setattr(warnings, "showwarning", handler)
        calc = make_calculator({"one": calculate})
        calc.compute()
        assert warnings.showwarning is handler
    assert len(visible) == 1 and visible[0][0:2] == ("visible", RuntimeWarning)
    report = calc.metric_reports["one"]
    assert [w.message for w in report.warnings] == ["visible"]
    assert (report.warnings[0].filename, report.warnings[0].lineno) == visible[0][2:]


def test_warning_promoted_to_exception_is_a_metric_failure(make_calculator):
    def calculate(data):
        warnings.warn("nonconverged", RuntimeWarning)
        pytest.fail("warning filter was ignored")
    calc = make_calculator({"one": calculate})
    events = []
    with warnings.catch_warnings(record=True) as visible:
        warnings.simplefilter("always")
        warnings.simplefilter("error", RuntimeWarning)
        calc.compute(on_metric_complete=lambda r, v: events.append((r, v)))
    report = calc.metric_reports["one"]
    assert report.status == "failed"
    assert report.error.type == "builtins.RuntimeWarning"
    assert report.error.message == "nonconverged"
    assert report.warnings == ()
    assert events == [(report, None)]
    assert len(visible) == 1 and "Caught" in str(visible[0].message)


def test_consumer_warnings_do_not_belong_to_the_metric(make_calculator):
    calc = make_calculator({"one": matrix})
    def consume(report, values):
        warnings.warn("consumer diagnostic", RuntimeWarning)
    with pytest.warns(RuntimeWarning, match="consumer diagnostic"):
        calc.compute(on_metric_complete=consume)
    assert calc.metric_reports["one"].warnings == ()
    assert calc.metric_reports["one"].status == "success"


def test_failure_report_does_not_retain_exception_arrays(make_calculator):
    references = []
    def calculate(data):
        working_array = np.zeros((32, 32))
        references.append(weakref.ref(working_array))
        raise ValueError("failed with a temporary array")
    calc = make_calculator({"one": calculate})
    with pytest.warns(UserWarning, match="Caught"):
        calc.compute()
    assert calc.metric_reports["one"].error.message == "failed with a temporary array"
    assert references[0]() is None


def test_failure_report_is_available_even_if_failure_warning_is_an_error(make_calculator):
    def calculate(data):
        raise ValueError("failed")
    calc = make_calculator({"one": calculate, "two": matrix})
    events = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="Caught"):
            calc.compute(on_metric_complete=lambda r, v: events.append((r, v)))
    assert list(calc.metric_reports) == ["one"]
    assert events == [(calc.metric_reports["one"], None)]
    assert calc.metric_reports["one"].error.type == "builtins.ValueError"


@pytest.mark.parametrize("result", [np.ones((2, 2)), np.ones((3, 3), dtype=int)])
def test_finalization_failure_is_not_reported_as_success(make_calculator, result):
    calc = make_calculator({"bad": lambda d: result.copy(), "after": matrix})
    events = []
    with pytest.warns(UserWarning, match="Caught"):
        calc.compute(on_metric_complete=lambda r, v: events.append((r, v)))
    assert [r.status for r, _ in events] == ["failed", "success"]
    assert events[0][1] is None
    assert calc.metric_reports["bad"].error.type == "builtins.ValueError"


def test_no_callback_does_not_copy_result_for_reporting(make_calculator, monkeypatch):
    calc = make_calculator({"one": matrix})
    original = pd.DataFrame.to_numpy
    def no_copy(self, *args, **kwargs):
        assert not kwargs.get("copy", False), "reporting copied a matrix without a consumer"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_numpy", no_copy)
    assert calc.compute() is None
    assert calc.metric_reports["one"].status == "success"


def test_one_channel_successful_nan_is_completed(make_calculator):
    calc = make_calculator({"one": matrix}, processes=1)
    events = []
    calc.compute(on_metric_complete=lambda r, v: events.append((r, v)))
    assert events[0][0].status == "success"
    assert events[0][1].shape == (1, 1)
    assert np.isnan(events[0][1]).all()


@pytest.mark.parametrize("invalid", [False, 1, "callback", []])
def test_bad_callback_is_rejected_before_calculation_or_report_reset(make_calculator, invalid):
    calc = make_calculator({"one": matrix})
    calc.compute()
    previous = calc.metric_reports["one"]
    with pytest.raises(TypeError, match="on_metric_complete"):
        calc.compute(on_metric_complete=invalid)
    assert calc.metric_reports["one"] is previous


@pytest.mark.parametrize("processes", [1, 2, 5])
@pytest.mark.parametrize("normalise,detrend", [(False, False), (False, True), (True, False), (True, True)])
def test_real_calculation_matches_default_with_callback(tmp_path, processes, normalise, detrend):
    config = tmp_path / "basic.yaml"
    config.write_text(yaml.safe_dump({".statistics.basic": {"Covariance": {
        "configs": [{"estimator": "EmpiricalCovariance"}, {"estimator": "LedoitWolf"}]}}}))
    raw = np.random.default_rng(74).normal(size=(processes, 64)) + np.linspace(0, 5, 64)
    default = Calculator(dataset=raw, configfile=str(config), normalise=normalise, detrend=detrend)
    observed = Calculator(dataset=raw, configfile=str(config), normalise=normalise, detrend=detrend)
    events = []
    with warnings.catch_warnings(record=True) as visible:
        warnings.simplefilter("always")
        assert default.compute() is None
        assert observed.compute(on_metric_complete=lambda r, v: events.append((r, v))) is None
    pd.testing.assert_frame_equal(observed.table, default.table, check_exact=True)
    np.testing.assert_array_equal(observed.dataset.to_numpy(), default.dataset.to_numpy())
    assert [r.identifier for r, _ in events] == list(observed.spis)
    for report, values in events:
        if processes == 1:
            # Existing Covariance squeezes a single process to 1D, which sklearn
            # rejects. This instrumentation preserves that failure, not fixes it.
            assert report.status == "failed" and values is None
            assert report.error.type == "builtins.ValueError"
            assert report.error.message.startswith("Expected 2D array")
            assert report.error.message == default.metric_reports[report.identifier].error.message
        else:
            assert report.status == "success"
            np.testing.assert_array_equal(values.view(np.uint64),
                                          default.table[report.identifier].to_numpy().view(np.uint64))
    assert len(visible) == (4 if processes == 1 else 0)
