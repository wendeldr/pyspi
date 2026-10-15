import json
import shutil

import numpy as np
import pytest
import yaml
import zarr

from pyspi.workflow import export_selection, load_config, open_run, preflight, run, WorkflowError
from pyspi.workflow._output import Journal, read_json
from pyspi.calculation import MetricReport


def test_legacy_export_is_read_only_and_maps_real_header_order(signal_edf, execution_config, tmp_path):
    source = signal_edf(labels=("EEG a'1", "eeg B2", "ECG ignored"))
    store = tmp_path / "patient.zarr"
    root = zarr.open_group(store, mode="w")
    root.create_array("patient_info/electrode", data=np.array(["B2", "A′1"]))
    before = {str(p.relative_to(store)): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    destination = tmp_path / "selection.json"
    result = export_selection(store, source, destination, patient_id="p", recording_id="baseline")
    assert result["recordings"][0]["channels"] == [{"index": 1, "name": "B2"}, {"index": 0, "name": "A′1"}]
    cfg = load_config(execution_config(source=source, metadata=str(destination)))
    assert cfg.recordings[0].patient_id == "p"
    assert cfg.recordings[0].recording_id == "baseline"
    assert preflight(cfg).recordings[0]["output_channels"] == ["B2", "A′1"]
    assert before == {str(p.relative_to(store)): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        export_selection(store, source, destination)
    with pytest.raises(WorkflowError, match="outside"):
        export_selection(store, source, store / "new.json")


@pytest.mark.parametrize("contacts,labels", [(["absent"], ("EEG A",)), (["A"], ("EEG A", "A")),
                                           (["A", "a"], ("EEG A",))])
def test_export_refuses_ambiguous_missing_or_duplicate_contacts(contacts, labels, signal_edf, tmp_path):
    source = signal_edf(labels=labels)
    store = tmp_path / "legacy.zarr"
    zarr.open_group(store, mode="w").create_array("patient_info/electrode", data=np.array(contacts))
    with pytest.raises(WorkflowError):
        export_selection(store, source, tmp_path / "new.json")
    assert not (tmp_path / "new.json").exists()


def test_directory_and_plain_file_list_inputs(signal_edf, execution_config, tmp_path):
    b, a = signal_edf(name="b.EDF"), signal_edf(name="a.edf")
    cfg = execution_config(source=a)
    doc = yaml.safe_load(cfg.read_text())
    doc.pop("recordings")
    doc["edf_directory"] = str(tmp_path)
    cfg.write_text(yaml.safe_dump(doc))
    assert [r.path for r in load_config(cfg).recordings] == [a, b]
    doc.pop("edf_directory")
    doc["recordings"] = [str(b), str(a)]
    cfg.write_text(yaml.safe_dump(doc))
    assert [r.path for r in load_config(cfg).recordings] == [b, a]


@pytest.mark.parametrize("key", ["sampling_rate_hz", "bandpass", "reference"])
def test_required_preparation_choices_cannot_be_omitted(key, execution_config):
    cfg = execution_config()
    doc = yaml.safe_load(cfg.read_text())
    doc.pop(key)
    cfg.write_text(yaml.safe_dump(doc))
    with pytest.raises(WorkflowError, match="required|reference"):
        load_config(cfg)


def test_relocation_partial_reads_and_successful_nan_availability(execution_config, tmp_path):
    [path] = run(execution_config(output_batch_mib=32/1024**2), max_windows=3)
    moved = tmp_path / "moved"
    shutil.copytree(path.parents[1], moved)
    reader = open_run(moved / "calculations" / path.name)
    assert (reader.path / reader.recording["metadata"]).resolve().is_file()
    manifest = read_json(reader.path / "run.json")
    first = reader.path / manifest["batches"][0]["path"] / "results.zarr"
    metric = reader.metrics[0]
    shutil.rmtree(first / metric)
    with pytest.warns(UserWarning):
        values, available = reader.load(metric, return_availability=True)
    assert np.isnan(values[:,:,0]).all() and not available[:,:,0].any()
    assert available[:,:,-1].all()
    # Losing a declaration must also warn, not silently supply unavailable NaNs.
    from pyspi.workflow._output import atomic_json
    manifest["batches"].pop(0)
    atomic_json(reader.path / "run.json", manifest)
    with pytest.warns(UserWarning, match="no batch declaration"):
        values, available = reader.load(metric, return_availability=True)
    assert not available[:,:,0].any() and available[:,:,-1].all()


def test_journal_keeps_failures_and_repairs_only_torn_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = Journal(path)
    journal.record(2, MetricReport("metric", "failed", .1, (), None))
    journal.close()
    with path.open("ab") as stream:
        stream.write(b'{"window":3')
    with pytest.warns(UserWarning, match="interrupted final"):
        journal = Journal(path)
    assert journal.latest[(2, "metric")]["report"]["status"] == "failed"
    journal.record(3, MetricReport("metric", "success", .2, (), None))
    journal.close()
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 2
    path.write_text('{"invalid": true}\n')
    with pytest.raises(WorkflowError, match="malformed diagnostic"):
        Journal(path)


def test_output_buffer_bound_is_independent_of_total_windows(execution_config, monkeypatch):
    import pyspi.workflow._run as runner
    batches = []
    original = runner._commit_batch
    def bounded(path, manifest, batch, buffers, *args):
        assert sum(a.nbytes for a in buffers.values()) <= 64
        assert batch["stop"] - batch["start"] <= 2
        batches.append((batch["start"], batch["stop"]))
        return original(path, manifest, batch, buffers, *args)
    monkeypatch.setattr(runner, "_commit_batch", bounded)
    run(execution_config(output_batch_mib=64/1024**2), max_windows=11)
    assert batches == [(0,2),(2,4),(4,6),(6,8),(8,10),(10,11)]


def test_run_lock_and_output_in_existing_zarr_rejected(execution_config, tmp_path):
    from pyspi.workflow._run import _run_lock
    path = tmp_path / "locked"
    with _run_lock(path):
        with pytest.raises(WorkflowError, match="another runner"):
            with _run_lock(path):
                pytest.fail("lock permitted a second writer")
    store = tmp_path / "old.zarr"
    zarr.open_group(store, mode="w")
    with pytest.raises(WorkflowError, match="outside existing Zarr"):
        run(execution_config(output_directory=str(store)), max_windows=1)


def test_resume_does_not_accept_a_deleted_committed_store(execution_config):
    cfg = execution_config()
    [path] = run(cfg, max_windows=1)
    batch = read_json(path / "run.json")["batches"][0]
    shutil.rmtree(path / batch["path"] / "results.zarr")
    with pytest.raises(WorkflowError, match="cannot resume unreadable"):
        run(cfg, resume=True, max_windows=1)
    from pyspi.workflow._output import atomic_json
    manifest = read_json(path / "run.json")
    manifest["batches"] = []
    atomic_json(path / "run.json", manifest)
    with pytest.raises(WorkflowError, match="declared window coverage"):
        run(cfg, resume=True, max_windows=1)
