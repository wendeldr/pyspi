"""Serial/spawned output equivalence and durable parent-owned checkpoints."""

import os

import numpy as np
import pytest
import yaml

from pyspi.workflow import load_config, open_run, run
from pyspi.workflow._output import read_json
from pyspi.workflow._parallel import numerical_threads


def test_parallel_matches_serial_and_resume_can_change_resources(signal_edf, execution_config, tmp_path):
    source = signal_edf(labels=("EEG A1", "EEG A2", "EEG B1", "EEG B2"))
    metrics = tmp_path / "parallel-metrics.yaml"
    metrics.write_text(yaml.safe_dump({
        ".statistics.basic": {"Covariance": {"configs": [{"squared": False}, {"squared": True}]}},
        ".statistics.spectral": {"SignedWeightedPhaseLagIndex": {"configs": [{}]}}}))
    results = []
    for workers in (1, 2):
        config = execution_config(source=source, metrics={"config": str(metrics)},
            run_name=f"workers-{workers}", output_batch_mib=.02,
            execution={"workers": workers, "threads_per_worker": 1, "spectral_cache_mib": 1})
        [path] = run(config, max_windows=4)
        reader = open_run(path)
        assert reader.recording["windows"]["step_samples"] == 128
        assert reader.recording["channels"] == ["EEG A1", "EEG A2", "EEG B1", "EEG B2"]
        results.append({metric: reader.load(metric) for metric in reader.metrics})
        assert all(set(reader.statuses(metric)) == {"success"} for metric in reader.metrics)
        session = read_json(path / "run.json")["sessions"][0]
        assert session["execution"]["workers"] == workers
        if workers == 2:
            assert session["worker_peak_rss_mib"]
            assert not any(session["worker_jvm_started"].values())
            assert all(int(pid) != os.getpid() for pid in session["worker_peak_rss_mib"])
            original_batches = read_json(path / "run.json")["batches"]
            document = yaml.safe_load(config.read_text())
            document["execution"] = {"workers": 1, "spectral_cache_mib": 2}
            config.write_text(yaml.safe_dump(document))
            run(config, resume=True, max_windows=4)
            assert read_json(path / "run.json")["batches"] == original_batches
    assert results[0].keys() == results[1].keys()
    for metric in results[0]:
        np.testing.assert_allclose(results[0][metric], results[1][metric], rtol=1e-12, atol=1e-14)


def test_parallel_failures_are_reported_and_retry_keeps_successes(signal_edf, execution_config, tmp_path):
    metrics = tmp_path / "failures.yaml"
    metrics.write_text(yaml.safe_dump({
        ".statistics.basic": {"Covariance": {"configs": [{}]}},
        ".statistics.spectral": {"GroupDelay": {"configs": [{"fmin": .12, "fmax": .12}]}}}))
    config = execution_config(metrics={"config": str(metrics)}, output_batch_mib=.01,
                              execution={"workers": 2})
    [path] = run(config, max_windows=2)
    reader = open_run(path)
    success = next(m for m in reader.metrics if m.startswith("cov_"))
    failed = next(m for m in reader.metrics if m.startswith("gd_"))
    assert set(reader.statuses(failed)) == {"failed"}
    assert all(e["report"]["error"] for e in reader.diagnostics(failed))
    before = reader.load(success).copy()
    run(config, resume=True, retry_failures=True, max_windows=2)
    np.testing.assert_array_equal(reader.load(success), before)
    assert set(reader.statuses(failed)) == {"failed"}


def test_thread_limits_restore_environment_on_error(monkeypatch):
    from threadpoolctl import threadpool_info
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    before = [(item["prefix"], item["num_threads"]) for item in threadpool_info()]
    with pytest.raises(RuntimeError):
        with numerical_threads(1):
            assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
            assert all(item["num_threads"] == 1 for item in threadpool_info())
            raise RuntimeError("controlled")
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert [(item["prefix"], item["num_threads"]) for item in threadpool_info()] == before


def test_parallel_interruption_leaves_resumable_uncommitted_batch(execution_config, monkeypatch):
    from pyspi.workflow._parallel import WindowPool
    original = WindowPool.map
    def interrupted(self, jobs):
        for result in original(self, jobs):
            yield result
            raise KeyboardInterrupt("controlled parent interruption")
    config = execution_config(execution={"workers": 2}, output_batch_mib=.01)
    monkeypatch.setattr(WindowPool, "map", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(config, max_windows=3)
    cfg = load_config(config)
    path = cfg.output_directory / cfg.recordings[0].patient_id / "calculations" / cfg.recordings[0].recording_id
    interrupted_manifest = read_json(path / "run.json")
    assert interrupted_manifest["state"] == "interrupted"
    assert all(batch["state"] == "pending" for batch in interrupted_manifest["batches"])
    monkeypatch.setattr(WindowPool, "map", original)
    run(config, resume=True, max_windows=3)
    reader = open_run(path)
    assert reader.recording["state"] == "complete"
    values, available = reader.load(reader.metrics[0], return_availability=True)
    assert available.all() and values.shape[-1] == 3


def test_profile_records_complete_run_and_memory(execution_config, tmp_path):
    from pyspi.workflow._profile import profile
    receipt = tmp_path / "profile.json"
    result = profile(execution_config(), max_windows=1, report_path=receipt)
    assert read_json(receipt) == result
    assert result["windows"] == 1 and result["runs"][0]["metrics"] == 1
    session = result["runs"][0]["session"]
    assert session["preparation_seconds"] > 0 and session["output_seconds"] > 0
    assert result["elapsed_seconds"] >= session["preparation_seconds"]
    assert result["parent_peak_rss_mib"] > 0
    assert result["process_tree_memory"]["samples"] > 0
