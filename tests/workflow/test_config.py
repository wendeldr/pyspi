from pathlib import Path

import pytest
import yaml

from pyspi.workflow import WorkflowError, load_config
from pyspi.workflow._config import read_yaml


def test_relative_paths_ids_and_defaults(edf, workflow, monkeypatch, tmp_path):
    source = edf()
    path = workflow(source, recordings=[{"path": source.name}],
                    metrics={"config": "metrics.yaml"}, output_directory="new_output")
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(path)
    assert cfg.recordings[0].path == source
    assert cfg.recordings[0].patient_id == cfg.recordings[0].recording_id == source.stem
    assert cfg.metric_config == tmp_path / "metrics.yaml"
    assert cfg.output_directory == tmp_path / "new_output"
    assert not cfg.output_directory.exists()
    assert cfg.detrend is cfg.zscore is True
    assert cfg.window_seconds == cfg.step_seconds == 1 / 2


@pytest.mark.parametrize("change,match", [
    ({"version": True}, "version"), ({"version": 2}, "version"),
    ({"unused": 1}, "unknown fields"), ({"recordings": []}, "at least"),
    ({"recordings": "file.EDF"}, "at least"),
    ({"recordings": [{"path": "x", "channels": []}]}, "nonempty list"),
    ({"recordings": [{"path": "x", "patient_id": 1}]}, "nonempty string"),
    ({"metrics": {"preset": "missing"}}, "unknown metrics preset"),
    ({"metrics": {"preset": "all", "config": "x"}}, "exactly one"),
    ({"metrics": None}, "mapping"), ({"metrics": {"config": ""}}, "nonempty string"),
    ({"window": {"length_seconds": 0}}, "positive finite"),
    ({"window": {"length_seconds": True}}, "positive finite"),
    ({"window": {"step_seconds": float("nan")}}, "positive finite"),
    ({"window": {"step_seconds": float("inf")}}, "positive finite"),
    ({"window": {"step_seconds": -1}}, "positive finite"),
    ({"window": {"overlap": 0}}, "unknown fields"),
    ({"sampling_rate_hz": "2048"}, "positive finite"),
    ({"sampling_rate_hz": False}, "positive finite"),
    ({"preprocessing": {"zscore": "false"}}, "booleans"),
    ({"preprocessing": {"detrend": 1}}, "booleans"),
    ({"preprocessing": {"rereference": "average"}}, "unknown fields"),
])
def test_invalid_config_rejected(change, match, edf, workflow):
    with pytest.raises(WorkflowError, match=match):
        load_config(workflow(edf(), **change))


@pytest.mark.parametrize("content", ["version: 1\nversion: 1\n", "metrics:\n  config: x\n  config: y\n"])
def test_duplicate_yaml_keys_fail(tmp_path, content):
    path = tmp_path / "duplicates.yaml"
    path.write_text(content)
    with pytest.raises(WorkflowError, match="duplicate YAML key"):
        read_yaml(path)


def test_invalid_yaml_and_missing_config(tmp_path):
    with pytest.raises(WorkflowError, match="cannot read"):
        load_config(tmp_path / "absent.yaml")
    path = tmp_path / "invalid.yaml"
    path.write_text("[unclosed")
    with pytest.raises(WorkflowError, match="cannot read"):
        read_yaml(path)


def test_glob_is_stable_and_relative(edf, workflow, tmp_path):
    b, a = edf(name="b.EDF"), edf(name="a.EDF")
    path = workflow(a)
    cfg = yaml.safe_load(path.read_text())
    cfg.pop("recordings")
    cfg["edf_glob"] = "*.EDF"
    path.write_text(yaml.safe_dump(cfg))
    assert [r.path for r in load_config(path).recordings] == [a, b]
    cfg["edf_glob"] = "no_match_*"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(WorkflowError, match="at least"):
        load_config(path)


def test_duplicate_identity_and_source(edf, workflow):
    a, b = edf(name="a.EDF"), edf(name="b.EDF")
    for records, match in [
        ([{"path": str(a)}, {"path": str(a), "recording_id": "other"}], "source selected"),
        ([{"path": str(a), "patient_id": "p", "recording_id": "r"},
          {"path": str(b), "patient_id": "p", "recording_id": "r"}], "duplicate patient"),
    ]:
        with pytest.raises(WorkflowError, match=match):
            load_config(workflow(a, recordings=records))


@pytest.mark.parametrize("preset", ["all", "fast", "sonnet", "fabfour", "reviewed"])
def test_presets_resolve_to_packaged_files(preset, edf, workflow):
    cfg = load_config(workflow(edf(), metrics={"preset": preset}))
    assert cfg.metric_config.is_file()


def test_required_rate_and_explicit_preprocessing_flags(edf, workflow):
    with pytest.raises(WorkflowError, match="sampling_rate_hz"):
        load_config(workflow(edf(), sampling_rate_hz=None))
    cfg = load_config(workflow(edf(), sampling_rate_hz=8,
                               window={"length_seconds": 2},
                               preprocessing={"detrend": False, "zscore": False}))
    assert cfg.sampling_rate_hz == 8 and cfg.window_seconds == cfg.step_seconds == 2
    assert not cfg.detrend and not cfg.zscore
