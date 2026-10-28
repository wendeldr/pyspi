"""Estimate arithmetic on hand-built calibrations; nothing here computes connectivity."""

import json
import math

import pytest

from pyspi.workflow import WorkflowError, load_config, preflight
from pyspi.workflow._estimate import (HEADROOM, calibration_issues, curve, estimate, estimate_recording,
                                      format_estimate, sidecar_path, window_seconds)
from pyspi.workflow._run import _batch_size, _environment, planned_windows
from pyspi.workflow.__main__ import main


def test_curve_is_exact_at_points_and_log_linear_between():
    points = [(8, 1.), (32, 4.)]
    assert curve(points, 8) == (1., "measured")
    assert curve(points, 32) == (4., "measured")
    seconds, mode = curve(points, 16)
    assert mode == "interpolated" and seconds == pytest.approx(2.)


def test_curve_clamps_below_and_extrapolates_with_bounded_exponent():
    assert curve([(8, 1.), (32, 4.)], 4) == (1., "clamped")
    assert curve([(8, 1.), (32, 4.)], 64) == (pytest.approx(8.), "extrapolated")
    assert curve([(8, 1.), (32, 256.)], 64)[0] == pytest.approx(256. * 2 ** 3)   # exponent 4 clamped to 3
    assert curve([(8, 4.), (32, 1.)], 64)[0] == pytest.approx(1.)                # negative clamped to flat
    assert curve([(8, 1.)], 16)[0] == pytest.approx(4.)                          # single point: quadratic
    assert curve([(8, 0.), (32, 4.)], 16)[0] == pytest.approx(4. / 3)            # zero time: linear fallback
    with pytest.raises(ValueError):
        curve([], 8)


def _calibration(metrics=("cov_empirical",), *, window=4, kind="edf", code=None,
                 threads=1, cache=256, detrend=True, zscore=True):
    points = []
    for channels, seconds in ((2, .01), (4, .04)):
        points.append({"channels": channels, "windows": 2, "preprocessing_seconds": .001,
                       "compute_seconds": seconds, "window_seconds": seconds + .001,
                       "metric_elapsed_sum_seconds": seconds, "warnings": 0,
                       "metrics": {m: {"seconds": seconds / len(metrics), "status": "success"} for m in metrics}})
    return {"schema": "pyspi-workflow-calibration/1", "metrics": list(metrics),
            "dtypes": {m: "<f8" for m in metrics}, "window_samples": window, "detrend": detrend, "zscore": zscore,
            "execution": {"workers": 1, "threads_per_worker": threads, "spectral_cache_mib": cache},
            "environment": {"pyspi_code_sha256": _environment()["pyspi_code_sha256"] if code is None else code},
            "source": {"kind": kind, "patient_id": "p", "recording_id": "r"},
            "calculator_setup_seconds": .5,
            "preparation": None if kind != "edf" else {"seconds": 2., "channels": 2, "native_samples": 100},
            "points": points,
            "output": {"seconds_per_batch": .1, "seconds_per_raw_byte": 1e-6,
                       "stored_ratio": {m: .5 for m in metrics}, "report_bytes_per_event": 100.}}


def test_window_seconds_reports_pooled_and_breakdown():
    result = window_seconds(_calibration(("a", "b")), 4)
    assert result["seconds"] == pytest.approx(.041) and result["mode"] == "measured"
    assert result["metrics"] == {"a": pytest.approx(.02), "b": pytest.approx(.02)}
    assert result["metric_sum_seconds"] == pytest.approx(.041)


def test_estimate_recording_arithmetic(edf, workflow):
    config = workflow(edf(samples=(8, 8), records=5), output_batch_mib=.0001, execution={"workers": 1})
    cfg = load_config(config)
    entry = preflight(cfg).recordings[0]
    calibration = _calibration()
    result = estimate_recording(entry, cfg, calibration)
    plan = planned_windows(entry, cfg)
    assert result["channels"] == 2 and result["windows"] == plan.count == 10
    batch = _batch_size(cfg, 2, 1)
    assert result["batch_windows"] == batch and result["batches"] == math.ceil(10 / batch)
    assert result["window_seconds"] == pytest.approx(.011) and result["mode"] == "measured"
    assert result["compute_seconds"] == pytest.approx(10 * .011 * HEADROOM["compute"])
    # 2 channels x 5 records x 8 samples native; calibration: 2 s per 100 channel-samples
    assert result["prepare_seconds"] == pytest.approx(2. / 100 * 80 * HEADROOM["prepare"])
    raw = 10 * 2 * 2 * 8
    assert result["raw_output_bytes"] == raw
    assert result["output_seconds"] == pytest.approx((result["batches"] * .1 + raw * 1e-6) * HEADROOM["output"])
    assert result["stored_bytes"] == pytest.approx((raw * .5 + 10 * 100) * HEADROOM["stored"])
    assert result["total_seconds"] == pytest.approx(result["prepare_seconds"] + result["compute_seconds"]
                                                    + result["output_seconds"])
    assert result["memory"]["prepared_bytes"] == 2 * 40 * 8
    clipped = estimate_recording(entry, cfg, calibration, max_windows=2)
    assert clipped["windows"] == 2 and clipped["compute_seconds"] == pytest.approx(2 * .011 * HEADROOM["compute"])


def test_windows_follow_duration_when_native_rate_differs(edf, workflow):
    # Native 8 Hz, target 16 Hz: preflight defers the plan; the runner derives it from duration.
    config = workflow(edf(samples=(8, 8), records=4), sampling_rate_hz=16, bandpass={"low_hz": .5, "high_hz": 6},
                      window={"length_seconds": .5, "step_seconds": .5})
    cfg = load_config(config)
    entry = preflight(cfg).recordings[0]
    assert entry["windows"] is None
    assert estimate_recording(entry, cfg, _calibration(window=8))["windows"] == 8


def test_parallel_workers_divide_compute_with_a_barrier_tail(edf, workflow):
    serial = load_config(workflow(edf(records=6), execution={"workers": 1}))
    parallel = load_config(workflow(edf(records=6), execution={"workers": 2}))
    entry = preflight(serial).recordings[0]
    one = estimate_recording(entry, serial, _calibration())
    two = estimate_recording(entry, parallel, _calibration())
    assert one["compute_seconds"] == pytest.approx(12 * .011 * HEADROOM["compute"])
    assert two["compute_seconds"] == pytest.approx((12 * .011 / (2 * .9) + two["batches"] * .011 / 4) * HEADROOM["compute"])
    assert two["compute_seconds"] < one["compute_seconds"]


def test_calibration_issues_distinguish_stale_from_notes(edf, workflow):
    cfg = load_config(workflow(edf()))
    report = preflight(cfg)
    assert calibration_issues(_calibration(), cfg, report) == ([], [])
    stale, notes = calibration_issues(_calibration(("other",), window=8, threads=2, detrend=False), cfg, report)
    assert len(stale) == 4 and notes == []
    stale, notes = calibration_issues(_calibration(kind="synthetic", code="0", cache=1024), cfg, report)
    assert stale == [] and len(notes) == 3


def test_estimate_uses_explicit_calibration_and_warns(edf, workflow, tmp_path):
    config = workflow(edf(samples=(8,) * 5, labels=("EEG A", "EEG B", "EEG C", "EEG D", "EEG E"), records=4))
    calibration = _calibration(kind="synthetic")
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(calibration))
    result = estimate(config, path)
    assert result["calibration"] == str(path) and result["recordings"][0]["channels"] == 5
    assert result["recordings"][0]["mode"] == "extrapolated"
    assert result["totals"]["prepare_seconds"] is None and result["totals"]["windows"] == 8
    assert result["totals"]["total_seconds"] == pytest.approx(.5 + result["recordings"][0]["total_seconds"])
    assert any("synthetic" in note for note in result["warnings"])
    assert any("exceed the calibrated 4 channels" in note for note in result["warnings"])
    text = format_estimate(result)
    assert text.startswith("recording") and "Warning:" in text and "extrapolated" in text
    assert "Calibration: synthetic data, p/r, channels 2, 4" in text
    stale = estimate(config, _calibration(("other",)))
    assert any("metric set differs" in note for note in stale["warnings"])
    with pytest.raises(WorkflowError):
        estimate(config, tmp_path / "missing.json")


def test_terminal_estimate_uses_current_sidecar(edf, workflow, capsys):
    config = workflow(edf(records=4))
    cfg = load_config(config)
    sidecar_path(cfg).write_text(json.dumps(_calibration()))
    assert main(["estimate", str(config), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["calibration"] == str(sidecar_path(cfg)) and result["warnings"] == []
    assert main(["estimate", str(config)]) == 0
    assert "Total:" in capsys.readouterr().out
