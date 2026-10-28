"""Calibration measurements and estimates on small real executions."""

import json
import subprocess
import sys

import pytest

from pyspi.workflow import load_config, open_run, preflight, run
from pyspi.workflow._calibrate import calibrate, default_ladder, pick_windows
from pyspi.workflow._estimate import estimate, sidecar_path
from pyspi.workflow._output import read_json


def test_default_ladder_and_window_choice():
    assert default_ladder(2) == [2] and default_ladder(4) == [2, 4]
    assert default_ladder(118) == [2, 4, 8, 16, 32, 64, 118]
    assert default_ladder(300) == [2, 4, 8, 16, 32, 64, 128, 300]
    assert pick_windows(40, 2) == [10, 29] and pick_windows(1, 3) == [0] and len(pick_windows(600, 3)) == 3


@pytest.fixture
def four_channel_config(execution_config, signal_edf):
    return execution_config(source=signal_edf(labels=("EEG A", "EEG B", "EEG C", "EEG D")))


def test_calibrate_measures_ladder_and_output(four_channel_config):
    config = four_channel_config
    result = calibrate(config, windows_per_point=2)
    assert read_json(sidecar_path(load_config(config))) == result
    assert result["schema"] == "pyspi-workflow-calibration/1"
    assert [p["channels"] for p in result["points"]] == [2, 4]
    assert all(p["window_seconds"] > 0 and p["windows"] == 2 for p in result["points"])
    assert result["metrics"] == ["cov_empirical"]
    assert result["dtypes"] == {"cov_empirical": "<f8"}
    assert result["preparation"]["seconds"] > 0 and result["preparation"]["channels"] == 4
    assert result["source"]["kind"] == "edf" and result["source"]["output_channels"] == 4
    output = result["output"]
    assert set(output["stored_ratio"]) == {"cov_empirical"} and output["stored_ratio"]["cov_empirical"] > 0
    assert output["seconds_per_batch"] >= 0 and output["seconds_per_raw_byte"] >= 0
    assert output["report_bytes_per_event"] > 0 and output["raw_bytes_per_window"] == 4 * 4 * 8
    assert result["calculator_setup_seconds"] > 0 and result["window_samples"] == 128


def test_estimate_auto_calibrates_then_reuses_sidecar(four_channel_config):
    config = four_channel_config
    cfg = load_config(config)
    assert not sidecar_path(cfg).exists()
    result = estimate(config)
    assert sidecar_path(cfg).is_file()
    assert any("measured a new calibration" in note for note in result["warnings"])
    recording = result["recordings"][0]
    assert recording["windows"] == preflight(cfg).recordings[0]["windows"].count == 40
    assert recording["mode"] == "measured" and recording["total_seconds"] > 0 and recording["stored_bytes"] > 0
    again = estimate(config)
    assert again["warnings"] == [] and again["calibration"] == str(sidecar_path(cfg))


def test_estimate_is_within_an_order_of_magnitude_of_a_bounded_run(four_channel_config):
    config = four_channel_config
    calibrate(config, windows_per_point=2)
    predicted = estimate(config, max_windows=4)["recordings"][0]
    [path] = run(config, max_windows=4)
    assert open_run(path).recording["state"] == "complete"
    session = read_json(path / "run.json")["sessions"][-1]
    measured = session["calculation_wall_seconds"] + session["output_seconds"]
    assert .2 * measured < predicted["compute_seconds"] + predicted["output_seconds"] < 5 * measured
    assert .2 * session["preparation_seconds"] < predicted["prepare_seconds"] < 5 * session["preparation_seconds"]


def test_synthetic_calibration_has_no_preparation(four_channel_config, tmp_path):
    result = calibrate(four_channel_config, out=tmp_path / "synthetic.json", channels=[3, 4], synthetic=True)
    assert result["preparation"] is None and result["source"]["kind"] == "synthetic"
    assert [p["channels"] for p in result["points"]] == [3, 4]
    estimated = estimate(four_channel_config, tmp_path / "synthetic.json")
    assert estimated["recordings"][0]["prepare_seconds"] is None
    assert any("synthetic" in note for note in estimated["warnings"])


def test_budget_stops_the_ladder_and_estimates_extrapolate(execution_config, signal_edf, tmp_path):
    config = execution_config(source=signal_edf(labels=("EEG A", "EEG B", "EEG C", "EEG D", "EEG E")))
    result = calibrate(config, out=tmp_path / "short.json", windows_per_point=1, budget_seconds=1e-6)
    assert result["ladder"] == [2, 4, 5] and [p["channels"] for p in result["points"]] == [2, 4]
    assert result["budget"]["stopped_before_channels"] == 5 and result["budget"]["predicted_seconds"] > 0
    assert result["output"]["channels"] == 4
    estimated = estimate(config, tmp_path / "short.json")
    assert estimated["recordings"][0]["mode"] == "extrapolated"
    assert any("stopped at its 0 s budget before 5 channels" in note for note in estimated["warnings"])


def test_terminal_calibrate_and_estimate(four_channel_config):
    config = four_channel_config
    result = subprocess.run([sys.executable, "-m", "pyspi.workflow", "calibrate", str(config),
                             "--channels", "2,all", "--windows-per-point", "1"],
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["channels"] == [2, 4]
    result = subprocess.run([sys.executable, "-m", "pyspi.workflow", "estimate", str(config), "--json"],
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["warnings"] == []
