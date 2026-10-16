import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.signal import detrend
from scipy.stats import zscore

from pyspi.workflow import run, open_run, WorkflowError, load_config
from pyspi.workflow._prepare import prepare_recording
from pyspi.workflow._output import read_json


def test_order_exact_roundtrip_and_independent_covariance(signal_edf, execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    source = signal_edf(labels=("EEG A", "EEG B", "EEG C", "ECG unused"))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    config = execution_config(source=source, reference="none", recordings=[
        {"path": str(source), "patient_id": "patient", "recording_id": "baseline",
         "channels": [{"index": i, "name": n} for i,n in [(2,"C"),(0,"A"),(1,"B")]]}],
        window={"length_seconds": .5, "step_seconds": .25})
    captured = []
    original = Covariance.multivariate
    def capture(self, data):
        result = original(self, data)
        expected = result.copy()
        np.fill_diagonal(expected, np.nan)
        captured.append(expected)
        return result
    monkeypatch.setattr(Covariance, "multivariate", capture)
    [path] = run(config, max_windows=5)
    reader = open_run(path)
    metric = reader.metrics[0]
    actual, availability = reader.load(metric, return_availability=True)
    assert actual.shape == (3,3,5) and availability.all()
    np.testing.assert_array_equal(actual.view(np.uint64), np.stack(captured, axis=2).view(np.uint64))
    cfg = load_config(config)
    prepared = prepare_recording(cfg.recordings[0], cfg)
    for window in range(5):
        samples = prepared.data[:, window*64:window*64+128]
        normalized = zscore(detrend(samples, axis=1), axis=1, ddof=1)
        expected = np.cov(normalized, bias=True)
        np.fill_diagonal(expected, np.nan)
        np.testing.assert_allclose(actual[:,:,window], expected, atol=1e-15, rtol=1e-15)
    np.testing.assert_array_equal(reader.load(metric, [4,0,4,-2]), actual[:,:,[4,0,4,3]])
    assert reader.load(metric, 1).shape == (3,3)
    assert reader.load(metric, []).shape == (3,3,0)
    for invalid in (True, .5, [False, 1]):
        with pytest.raises(TypeError):
            reader.load(metric, invalid)
    with pytest.raises(IndexError):
        reader.load(metric, [5])
    assert reader.recording["channels"] == ["C","A","B"]
    assert reader.recording["windows"]["step_samples"] == 64
    assert (path / reader.recording["metadata"]).resolve().is_file()
    assert set(reader.statuses(metric)) == {"success"}
    events = list(reader.diagnostics(metric))
    assert [e["window"] for e in events] == list(range(5))
    assert all(e["report"]["elapsed_seconds"] >= 0 for e in events)
    session = read_json(path / "run.json")["sessions"][0]
    assert all(session[key] > 0 for key in ("preparation_seconds", "output_seconds", "elapsed_seconds", "metric_elapsed_seconds"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(WorkflowError, match="already exists"):
        run(config, max_windows=5)


def test_interruption_reuses_batches_and_preserves_random_draws(execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    calls, seen = [], []
    def stochastic(self, data):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        value = np.random.random()
        seen.append(value)
        return np.full((data.n_processes, data.n_processes), value)
    monkeypatch.setattr(Covariance, "multivariate", stochastic)
    config = execution_config(reference="none", output_batch_mib=64/1024**2)
    np.random.seed(777)
    with pytest.raises(KeyboardInterrupt):
        run(config, max_windows=5)
    path = load_config(config).output_directory / "signals" / "calculations" / "signals"
    reader = open_run(path)
    metric = reader.metrics[0]
    with pytest.warns(UserWarning, match="unfinished"):
        partial, good = reader.load(metric, return_availability=True)
    assert good[:,:,:2].all() and not good[:,:,2:].any()
    saved = partial[:,:,:2].copy()
    assert list(reader.statuses(metric)) == ["success", "success", "unfinished", "unfinished", "unfinished"]
    run(config, resume=True, max_windows=5)
    np.testing.assert_array_equal(open_run(path).load(metric)[:,:,:2].view(np.uint64), saved.view(np.uint64))
    assert len(calls) == 6  # two retained successes, one interruption, three new successes
    np.testing.assert_array_equal(seen, np.random.RandomState(777).random_sample(5))


def test_pending_failures_skip_automatically_and_retry_only_explicitly(execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    calls = []
    def fail_then_interrupt(self, data):
        calls.append(1)
        if len(calls) == 1:
            raise ArithmeticError("controlled calculation failure")
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return np.full((data.n_processes, data.n_processes), np.nan)
    monkeypatch.setattr(Covariance, "multivariate", fail_then_interrupt)
    config = execution_config(output_batch_mib=64/1024**2)
    with pytest.raises(KeyboardInterrupt):
        run(config, max_windows=2)
    [path] = run(config, resume=True, max_windows=2)
    reader = open_run(path)
    metric = reader.metrics[0]
    assert len(calls) == 3  # failed first window is skipped
    assert list(reader.statuses(metric)) == ["failed", "success"]
    values, good = reader.load(metric, return_availability=True)
    assert np.isnan(values).all() and good.all()  # both are durably stored NaNs
    run(config, resume=True, max_windows=2)
    assert len(calls) == 3
    run(config, resume=True, retry_failures=True, max_windows=2)
    assert len(calls) == 4
    assert list(open_run(path).statuses(metric)) == ["success", "success"]
    with pytest.raises(WorkflowError, match="requires explicit resume"):
        run(config, retry_failures=True)


@pytest.mark.parametrize("change", ["source", "window", "reference", "metrics", "target", "limit"])
def test_incompatible_resume_rejected(change, execution_config, metric_config):
    import yaml
    config = execution_config()
    run(config, max_windows=1)
    cfg = yaml.safe_load(config.read_text())
    if change == "source":
        source = Path(cfg["recordings"][0]["path"])
        content = bytearray(source.read_bytes())
        content[-1] ^= 1
        source.write_bytes(content)
    elif change == "window": cfg["window"] = {"length_seconds": 1}
    elif change == "reference": cfg["reference"] = "none"
    elif change == "metrics": cfg["metrics"] = {"config": str(metric_config({"Precision": {"configs": [{}]}}))}
    elif change == "target": cfg["sampling_rate_hz"] = 512
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(WorkflowError, match="incompatible resume"):
        run(config, resume=True, max_windows=2 if change == "limit" else 1)


def test_two_metrics_share_only_current_window_cache(execution_config, metric_config, monkeypatch):
    from pyspi.statistics.basic import Covariance, Precision
    seen = []
    def first(self, data):
        assert not hasattr(data, "test_cache")
        data.test_cache = object()
        seen.append(data.test_cache)
        return np.eye(data.n_processes)
    def second(self, data):
        assert data.test_cache is seen[-1]
        return np.eye(data.n_processes)
    monkeypatch.setattr(Covariance, "multivariate", first)
    monkeypatch.setattr(Precision, "multivariate", second)
    metrics = metric_config({"Covariance": {"configs": [{}]}, "Precision": {"configs": [{}]}})
    [path] = run(execution_config(metrics={"config": str(metrics)}), max_windows=4)
    assert len(seen) == 4 and len({id(c) for c in seen}) == 4
    reader = open_run(path)
    assert len(reader.metrics) == 2
    # Selective loading must never touch another metric's arrays.
    import pyspi.storage._store as storage
    original = storage.Results.load
    def only(self, metric, *args, **kwargs):
        assert metric == reader.metrics[0]
        return original(self, metric, *args, **kwargs)
    monkeypatch.setattr(storage.Results, "load", only)
    assert reader.load(reader.metrics[0]).shape == (2,2,4)


def test_real_terminal_execution(execution_config):
    config = execution_config()
    result = subprocess.run([sys.executable, "-m", "pyspi.workflow", "run", str(config), "--max-windows", "1"],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)["runs"]
    assert open_run(paths[0]).recording["state"] == "complete"


def test_constructor_label_state_does_not_break_same_process_resume(execution_config, metric_config):
    from pyspi.statistics.basic import SpearmanR
    before = SpearmanR.labels.copy()
    metrics = metric_config({"SpearmanR": {"configs": [{"squared": False}]}})
    config = execution_config(metrics={"config": str(metrics)})
    [path] = run(config, max_windows=2)
    assert SpearmanR.labels == before
    run(config, resume=True, max_windows=2)
    assert SpearmanR.labels == before
    assert open_run(path).recording["state"] == "complete"
    # Existing code outside the runner can mutate these same class lists.
    SpearmanR.labels.append("incidental-existing-label")
    try:
        run(config, resume=True, max_windows=2)
    finally:
        SpearmanR.labels[:] = before


def test_failed_first_batch_does_not_force_float64_precision(execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    count = 0
    def metric(self, data):
        nonlocal count
        count += 1
        if count == 1:
            raise ValueError("first window fails")
        bits = np.array([[0, 0x80000000], [0x7fc00123, 0]], dtype=np.uint32)
        return bits.view(np.float32)
    monkeypatch.setattr(Covariance, "multivariate", metric)
    [path] = run(execution_config(output_batch_mib=32/1024**2), max_windows=2)
    reader = open_run(path)
    values = reader.load(reader.metrics[0])
    assert values.dtype == np.float32
    assert values[0,1,1].view(np.uint32) == 0x80000000
    assert values[1,0,1].view(np.uint32) == 0x7fc00123
    assert list(reader.statuses(reader.metrics[0])) == ["failed", "success"]


def test_window_preprocessing_failure_is_recorded_not_a_success(execution_config, edf):
    source = edf(samples=(256,256), records=10)
    [path] = run(execution_config(source=source), max_windows=2)
    reader = open_run(path)
    events = list(reader.diagnostics())
    assert all(e["stage"] == "window_preprocessing" and e["report"]["status"] == "failed" for e in events)
    assert all("zero or numerically unusable variance" in e["report"]["error"]["message"] for e in events)
    assert np.isnan(reader.load(reader.metrics[0])).all()


def test_failed_batch_publication_does_not_claim_completion(execution_config, monkeypatch):
    import pyspi.workflow._run as runner
    original = runner.atomic_json
    triggered = False
    def fail_publication(path, manifest):
        nonlocal triggered
        if path.name == "run.json" and not triggered and any(b["state"] == "complete" for b in manifest["batches"]):
            triggered = True
            raise OSError("controlled commit failure")
        return original(path, manifest)
    monkeypatch.setattr(runner, "atomic_json", fail_publication)
    config = execution_config()
    with pytest.raises(OSError, match="controlled commit"):
        run(config, max_windows=2)
    path = load_config(config).output_directory / "signals" / "calculations" / "signals"
    manifest = read_json(path / "run.json")
    assert manifest["state"] == "failed" and manifest["batches"][0]["state"] == "pending"
    run(config, resume=True, max_windows=2)
    assert open_run(path).recording["state"] == "complete"


def test_interrupted_explicit_retry_keeps_old_generation(execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    monkeypatch.setattr(Covariance, "multivariate", lambda *a: (_ for _ in ()).throw(ValueError("failure")))
    config = execution_config()
    [path] = run(config, max_windows=2)
    old = read_json(path / "run.json")["batches"][0]["path"]
    monkeypatch.setattr(Covariance, "multivariate", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run(config, resume=True, retry_failures=True, max_windows=2)
    assert read_json(path / "run.json")["batches"][0]["path"] == old
    assert list(open_run(path).statuses(open_run(path).metrics[0])) == ["failed", "failed"]


def test_cohort_resume_starts_unvisited_recordings_and_keeps_order(signal_edf, execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    sources = [signal_edf(name=f"{name}.EDF") for name in ("b", "a", "c")]
    calls = 0
    original = Covariance.multivariate
    def interrupt(self, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        return original(self, data)
    monkeypatch.setattr(Covariance, "multivariate", interrupt)
    config = execution_config(source=sources[0], recordings=[str(p) for p in sources])
    with pytest.raises(WorkflowError, match="no run to resume"):
        run(config, resume=True, max_windows=2)
    with pytest.raises(KeyboardInterrupt):
        run(config, max_windows=2)
    paths = run(config, resume=True, max_windows=2)
    assert [open_run(p).recording["patient_id"] for p in paths] == ["b", "a", "c"]
    assert all(open_run(p).recording["state"] == "complete" for p in paths)
    assert calls == 7


def test_resume_identifies_session_without_final_report(execution_config):
    from pyspi.workflow._output import atomic_json
    config = execution_config()
    [path] = run(config, max_windows=1)
    manifest = read_json(path / "run.json")
    manifest["state"] = "running"
    manifest["sessions"][0].update(status="running", elapsed_seconds=0.)
    atomic_json(path / "run.json", manifest)
    run(config, resume=True, max_windows=1)
    previous = read_json(path / "run.json")["sessions"][0]
    assert previous["status"] == "interrupted" and previous["elapsed_seconds"] is None
    assert previous["ended_without_final_report"] is True


def test_module_alias_groups_preserve_requested_metric_order(execution_config, tmp_path):
    import yaml
    from pyspi.workflow import preflight
    metrics = tmp_path / "aliases.yaml"
    metrics.write_text(yaml.safe_dump({
        ".statistics.basic": {"Covariance": {"configs": [{}]}, "Precision": {"configs": [{}]}},
        "pyspi.statistics.basic": {"Covariance": {"configs": [{"squared": True}]}}}, sort_keys=False))
    config = execution_config(metrics={"config": str(metrics)})
    expected = [m.identifier for m in preflight(config).metrics]
    [path] = run(config, max_windows=1)
    assert open_run(path).metrics == expected
    assert [e["report"]["identifier"] for e in open_run(path).diagnostics()] == expected
