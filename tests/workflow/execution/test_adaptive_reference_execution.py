"""Patient-specific reference resolution survives execution and shared output."""
from dataclasses import replace

import numpy as np
import pytest

from seeg_dataset import create_patient, open_run
from pyspi.workflow import load_config, preflight, run, WorkflowError
from pyspi.workflow._prepare import prepare_recording
from pyspi.workflow._dataset import dataset_address


def setup(execution_config, signal_edf, tmp_path):
    root = tmp_path / "lake"
    recordings = []
    for pid, names, groups in [("first", ["A4", "A1", "A3"], ["A"] * 3),
                                ("second", ["B2", "B3", "B1", "C1"], ["B"] * 3 + ["C"])]:
        # No coordinates: equal shaft spacing determines fractional weights.
        create_patient(root / f"{pid}.zarr", pid, arrays={"electrode": names, "electrodegroup": groups})
        source = signal_edf(labels=["EEG " + n for n in reversed(names)], name=f"{pid}.edf")
        recordings.append({"path": str(source), "patient_id": pid, "recording_id": "baseline"})
    path = execution_config(recordings=recordings, dataset={"directory": str(root)}, output_directory=None,
                            reference="adaptive_local", preprocessing={"zscore": False, "detrend": False})
    return load_config(path), root


@pytest.mark.parametrize("workers", [1, 2])
def test_patient_specific_axes_rules_serial_and_spawned_output(workers, execution_config, signal_edf, tmp_path):
    cfg, root = setup(execution_config, signal_edf, tmp_path)
    cfg = replace(cfg, execution={**cfg.execution, "workers": workers})
    report = preflight(cfg)
    assert report.recordings[0]["output_channels"] == ["A4", "A1", "A3"]
    assert report.recordings[1]["output_channels"] == ["B2", "B3", "B1", "C1"]
    assert all(all(e["reference_available"]) for e in report.recordings)
    assert report.recordings[1]["reference"]["rules"]["C1"]["rule"] == "car"
    paths = run(cfg, max_windows=1)
    for path, entry in zip(paths, report.recordings):
        reader = open_run(path)
        assert reader.recording["channels"] == entry["output_channels"]
        assert reader.recording["reference"] == entry["reference"]
        assert "adaptive_local" in path.parts
        values = reader.load(reader.metrics[0])
        assert np.isfinite(values[:, :, 0][~np.eye(len(values), dtype=bool)]).all()
        assert list(reader.statuses(reader.metrics[0])) == ["success"]
    assert run(cfg, max_windows=1, resume=True) == paths
    car_report = preflight(replace(cfg, reference={"method": "car"}))
    assert [e["output_channels"] for e in report.recordings] == [e["output_channels"] for e in car_report.recordings]


def test_continuous_preparation_matches_independent_formula(execution_config, signal_edf, tmp_path):
    cfg, _ = setup(execution_config, signal_edf, tmp_path)
    report = preflight(cfg)
    recording = report.configuration.recordings[0]
    source = prepare_recording(recording, replace(cfg, reference={"method": "none"}))
    actual = prepare_recording(recording, cfg)
    # Stored order A4, A1, A3; independently specify all three outputs.
    expected = np.array([source.data[0] - source.data[2], source.data[1] - source.data[2],
                         source.data[2] - source.data[1]/3 - source.data[0]*2/3])
    np.testing.assert_allclose(actual.data, expected, atol=1e-22)
    assert actual.channel_names == source.channel_names


def test_changed_patient_layout_changes_address_and_blocks_explicit_run_resume(execution_config, signal_edf, tmp_path):
    import zarr
    root = tmp_path / "lake"
    patient = root / "example.zarr"
    names = ["X11", "X12", "X13"]
    create_patient(patient, "example", arrays={"electrode": names, "electrodegroup": ["X"] * 3})
    source = signal_edf(labels=["EEG " + n for n in names])
    cfg = load_config(execution_config(source=source, recordings=[{"path": str(source), "patient_id": "example",
        "recording_id": "baseline"}], dataset={"directory": str(root)}, output_directory=None,
        reference="adaptive_local"))
    before = preflight(cfg)
    old_address = dataset_address(cfg, cfg.recordings[0], before.recordings[0])[1]
    explicit = replace(cfg, run_name="same")
    run(explicit, max_windows=1)
    # Digit-containing shaft names are valid: authoritative metadata resolves
    # X + contacts 11..13 versus X1 + contacts 1..3. Changed provenance blocks reuse.
    zarr.open_group(patient, mode="r+")["patient_info/electrodegroup"][:] = ["X1"] * 3
    after = preflight(cfg)
    new_address = dataset_address(cfg, cfg.recordings[0], after.recordings[0])[1]
    assert old_address != new_address
    with pytest.raises(WorkflowError, match="incompatible resume"):
        run(explicit, max_windows=1, resume=True)


def test_calibration_accepts_resolved_reference(execution_config, signal_edf, tmp_path):
    from pyspi.workflow._calibrate import calibrate
    from pyspi.workflow._estimate import estimate
    cfg, _ = setup(execution_config, signal_edf, tmp_path)
    measured = calibrate(cfg, windows_per_point=1, channels=[2, 4])
    assert measured["preparation"]["details"]["reference"]["patient_id"] == "second"
    estimated = estimate(cfg, measured)
    assert not any("reference differs" in note for note in estimated["warnings"])
