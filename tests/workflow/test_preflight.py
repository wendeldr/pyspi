from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pyspi.workflow import IncompleteWindowWarning, WorkflowError, preflight
from pyspi.workflow.__main__ import main
from pyspi.workflow._windows import plan_windows


@pytest.mark.parametrize("samples,length,step,expected,tail", [
    (12, 4, 4, [(0, 4), (4, 8), (8, 12)], False),
    (12, 4, 2, [(0, 4), (2, 6), (4, 8), (6, 10), (8, 12)], False),
    (13, 4, 4, [(0, 4), (4, 8), (8, 12)], True),
    (3, 4, 4, [], True), (0, 4, 4, [], False),
    (12, 4, 8, [(0, 4), (8, 12)], False),
])
def test_exact_sample_ranges(samples, length, step, expected, tail):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = plan_windows(samples, Fraction(8), Fraction(length, 8), Fraction(step, 8))
    assert list(result.ranges()) == expected
    assert result.as_dict()["sample_ranges"] == expected
    assert "sample_ranges" not in result.as_dict(include_ranges=False)
    assert len(caught) == int(tail)
    if tail:
        assert caught[0].category is IncompleteWindowWarning
    assert "tail" not in str(result.as_dict()).lower()


def test_ready_report_order_and_inputs_unchanged(edf, workflow):
    source = edf(labels=("EEG A1", "EEG B2", "ECG ECG"), samples=(8, 8, 16))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    path = workflow(source, recordings=[{"path": str(source), "patient_id": "patient",
                                        "recording_id": "baseline", "channels": [
                                            {"label": "EEG B2", "name": "B2"},
                                            {"label": "EEG A1", "name": "A1"}]}],
                    output_directory="uncreated-output")
    report = preflight(path)
    data = report.as_dict()
    assert report.ready and data["sampling_rate_hz"] == 8
    assert data["preprocessing"] == ["detrend", "zscore"]
    entry = data["recordings"][0]
    assert entry["patient_id"] == "patient" and entry["recording_id"] == "baseline"
    assert [c["source_index"] for c in entry["channels"]] == [1, 0]
    assert [c["name"] for c in entry["channels"]] == ["B2", "A1"]
    assert entry["windows"]["sample_ranges"] == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    assert json.loads(json.dumps(data))["ready"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not Path(data["output_directory"]).exists()


@pytest.mark.parametrize("native", [4, 16])
def test_upsampling_and_downsampling_needed_no_target_ranges(native, edf, workflow):
    source = edf(samples=(native, native))
    report = preflight(workflow(source))
    entry = report.as_dict()["recordings"][0]
    assert not report.ready and entry["status"] == "requires_resampling"
    assert entry["windows"] is None
    assert {c["native_sampling_rate_hz"] for c in entry["channels"]} == {native}


def test_mixed_native_rates_are_seen_before_reader_resampling(edf, workflow):
    report = preflight(workflow(edf(samples=(8, 16))))
    assert not report.ready
    assert [c["native_sampling_rate_hz"] for c in report.recordings[0]["channels"]] == [8, 16]


def test_unselected_and_annotation_rates_do_not_force_resampling(edf, workflow):
    source = edf(labels=("EEG A", "EEG B", "EDF Annotations"), samples=(8, 16, 20))
    assert preflight(workflow(source, recordings=[{"path": str(source), "channels": [0]}])).ready


def test_rate_required_even_if_all_selected_signals_agree(edf, workflow):
    with pytest.raises(WorkflowError, match="sampling_rate_hz"):
        preflight(workflow(edf(), sampling_rate_hz=None))
    with pytest.raises(WorkflowError, match="sampling_rate_hz"):
        preflight(workflow(edf(samples=(8, 16)), sampling_rate_hz=None))


def test_rates_checked_across_recordings(edf, workflow):
    a, b = edf(name="a.EDF"), edf(name="b.EDF", samples=(16, 16))
    path = workflow(a, recordings=[{"path": str(a)}, {"path": str(b)}])
    assert [r["status"] for r in preflight(path).recordings] == ["ready", "requires_resampling"]
    with pytest.raises(WorkflowError, match="sampling_rate_hz"):
        preflight(workflow(a, recordings=[{"path": str(a)}, {"path": str(b)}], sampling_rate_hz=None))


def test_channel_error_identifies_recording(edf, workflow):
    source = edf()
    cfg = workflow(source, recordings=[{"path": str(source), "patient_id": "p",
                                        "recording_id": "baseline", "channels": ["absent"]}])
    with pytest.raises(WorkflowError, match=r"p/baseline.*source.EDF.*absent"):
        preflight(cfg)


def test_tail_warnings_identify_patients_with_shared_recording_id(edf, workflow):
    a, b = edf(name="a.EDF", samples=(9, 9)), edf(name="b.EDF", samples=(9, 9))
    # One 2-second window leaves a 1-second tail in each 3-second recording.
    cfg = workflow(a, sampling_rate_hz=9, window={"length_seconds": 2, "step_seconds": 2},
                   recordings=[{"path": str(a), "patient_id": "p1", "recording_id": "baseline"},
                               {"path": str(b), "patient_id": "p2", "recording_id": "baseline"}])
    with pytest.warns(IncompleteWindowWarning) as caught:
        preflight(cfg)
    assert len(caught) == 2
    assert "p1/baseline" in str(caught[0].message) and "p2/baseline" in str(caught[1].message)


@pytest.mark.parametrize("window", [{"length_seconds": .1}, {"step_seconds": .1}])
def test_nonintegral_window_grid_rejected_even_before_resampling(edf, workflow, window):
    with pytest.raises(WorkflowError, match="whole number of samples"):
        preflight(workflow(edf(samples=(16, 16)), window=window))


def test_fractional_native_rate_and_sample_grid(edf, workflow):
    source = edf(samples=(1, 1), duration="2", records=3)
    report = preflight(workflow(source, sampling_rate_hz=.5,
                               bandpass={"low_hz": .025, "high_hz": .1},
                               window={"length_seconds": 2, "step_seconds": 2}))
    assert report.sampling_rate_hz == Fraction(1, 2)
    assert list(report.recordings[0]["windows"].ranges()) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize("length,step", [(Fraction(1, 10), Fraction(1)), (Fraction(0), Fraction(1))])
def test_window_helper_rejects_invalid_sample_counts(length, step):
    with pytest.raises(WorkflowError):
        plan_windows(20, Fraction(8), length, step)


@pytest.mark.parametrize("count", [1, 2000])
def test_no_arbitrary_channel_count_bound(count, edf, workflow):
    source = edf(labels=tuple(f"EEG ch{i}" for i in range(count)), samples=(8,) * count)
    report = preflight(workflow(source))
    assert report.ready and len(report.recordings[0]["channels"]) == count


def test_cli_reports_json_and_exit_states(edf, workflow, capsys):
    cfg = workflow(edf())
    assert main(["preflight", str(cfg), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["recordings"][0]["windows"]["sample_ranges"]) == 6
    cfg = workflow(edf(samples=(16, 16)))
    assert main(["preflight", str(cfg)]) == 1
    assert "requires resampling" in capsys.readouterr().out
    assert main(["preflight", str(cfg.parent / "absent.yaml")]) == 2
    assert "Preflight error" in capsys.readouterr().err


def test_real_module_command_is_read_only(edf, workflow):
    source = edf()
    cfg = workflow(source)
    before = source.read_bytes()
    run = subprocess.run([sys.executable, "-m", "pyspi.workflow", "preflight", str(cfg), "--json"],
                         text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    data = json.loads(run.stdout)
    assert data["ready"] and data["recordings"][0]["windows"]["count"] == 6
    assert source.read_bytes() == before
