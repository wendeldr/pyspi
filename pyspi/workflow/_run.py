"""EDF execution with bounded window workers and parent-owned checkpoints."""

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import importlib
from importlib.metadata import requires, version
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
from uuid import uuid4
import warnings

import numpy as np
import yaml

from ._config import WorkflowError
from ._edf import read_header
from ._output import RUN_SCHEMA, Journal, atomic_json, child_path, read_json
from ._preflight import preflight
from ._prepare import prepare_recording
from ._windows import plan_windows
from ._precision import resolve_dtypes, convert_output


@contextmanager
def _run_lock(path):
    import fcntl

    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError(f"another runner is writing {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _source_identity(path):
    header = read_header(path)
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.sha256(stream.read(256 + 256 * len(header.signals))).hexdigest()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "header_sha256": digest}


def _environment():
    from packaging.requirements import Requirement

    packages = {"pyspi"} | {Requirement(r).name for r in requires("pyspi") or [] if not Requirement(r).marker}
    packages |= {"zarr", "numcodecs", "pcodec", "google-crc32c"}
    versions = {name: version(name) for name in sorted(packages)}
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for source in sorted(root.rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return {"versions": versions, "pyspi_code_sha256": digest.hexdigest(),
            "python": sys.version.split()[0]}


def _contract(cfg, recording, entry, metrics, environment, max_windows):
    contract = {"patient_id": recording.patient_id, "recording_id": recording.recording_id,
                "source": _source_identity(recording.path), "channels": entry["channels"],
                "sampling_rate_hz": float(cfg.sampling_rate_hz), "bandpass": list(cfg.bandpass),
                "notch": cfg.notch, "reference": cfg.reference,
                "window_seconds": float(cfg.window_seconds), "step_seconds": float(cfg.step_seconds),
                "detrend": cfg.detrend, "zscore": cfg.zscore,
                "metrics": [m.as_dict() for m in metrics], "environment": environment,
                "output_batch_mib": cfg.output_batch_mib, "max_windows": max_windows,
                "output_dtypes": resolve_dtypes(cfg, [m.identifier for m in metrics])}
    if cfg.reference["method"] == "rest":
        forward = Path(cfg.reference["forward"])
        stat = forward.stat()
        contract["reference_forward"] = {"path": str(forward), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # JSON normalizes tuple-valued descriptions before comparison on resume.
    import json
    return json.loads(json.dumps(contract, allow_nan=False))


def _compatibility_fields(contract):
    # Scientific label state is known to be mutable in the current estimators;
    # constructor warning visibility also depends on the caller's filters.
    # Neither controls numerical calculation. Preserve the first run's labels
    # as metadata without making these incidental descriptions resume gates.
    if not isinstance(contract, dict) or not isinstance(contract.get("metrics"), list):
        return None
    return {**contract, "metrics": [{k: v for k, v in m.items()
             if k not in ("labels", "initialization_warnings")} for m in contract["metrics"]]}


def _calculator(metrics, cfg, directory):
    """Normalize null/default requests once; then use the existing Calculator."""
    from pyspi.calculator import Calculator

    document = {}
    for metric in metrics:
        spec = document.setdefault(metric.module, {}).setdefault(metric.class_name,
                 {"dependencies": list(metric.dependencies), "configs": []})
        spec["configs"].append(metric.parameters)
    filename = directory / "metrics.yaml"
    filename.write_text(yaml.safe_dump(document, sort_keys=False))
    with redirect_stdout(sys.stderr):
        calc = Calculator(configfile=str(filename), detrend=cfg.detrend, normalise=cfg.zscore,
                          spectral_cache_mib=cfg.execution["spectral_cache_mib"])
    identifiers = [m.identifier for m in metrics]
    if set(calc.spis) != set(identifiers):
        raise WorkflowError("Calculator initialized a different metric inventory from preflight")
    # Relative and absolute YAML module aliases can describe the same class in
    # separate groups. Canonical construction merges those groups; restore the
    # requested compute/output order explicitly after checking the inventory.
    ordered = {identifier: calc.spis[identifier] for identifier in identifiers}
    calc.spis.clear()
    calc.spis.update(ordered)
    return calc


@contextmanager
def _class_label_state(metrics):
    """Keep constructor label-list side effects inside this invocation.

    This mirrors preflight's isolation; it does not correct scientific labels
    or change the instances used for calculation.
    """
    lists = {}
    for metric in metrics:
        cls = getattr(importlib.import_module(metric.module), metric.class_name)
        for parent in cls.__mro__:
            labels = vars(parent).get("labels")
            if isinstance(labels, list):
                lists.setdefault(id(labels), (labels, labels.copy()))
    try:
        yield
    finally:
        for labels, original in lists.values():
            labels[:] = original


def _batch_size(cfg, n, metrics):
    # Integer counts are used only by estimates without per-metric inventory.
    if isinstance(metrics, int):
        bytes_per_pair = np.dtype(cfg.output_dtype).itemsize * metrics
    else:
        bytes_per_pair = sum(np.dtype(dt).itemsize for dt in resolve_dtypes(cfg, metrics).values())
    one = n * n * bytes_per_pair
    target = int(cfg.output_batch_mib * 1024 ** 2)
    if one > target:
        warnings.warn(f"one output window needs approximately {one} bytes, exceeding output_batch_mib; "
                      "using one window per batch", UserWarning, stacklevel=2)
    return max(1, target // one)


def planned_windows(entry, cfg):
    """The runner's window plan for a preflight recording entry, before any preparation."""
    # MNE's polyphase output uses round(native_samples * target/native_rate).
    samples = max(1, round(entry["duration_seconds"] * float(cfg.sampling_rate_hz)))
    return plan_windows(samples, cfg.sampling_rate_hz, cfg.window_seconds, cfg.step_seconds,
                        recording=f"{entry['patient_id']}/{entry['recording_id']}")


def _new_manifest(contract, entry, cfg, metadata_link):
    plan = planned_windows(entry, cfg)
    if contract["max_windows"] is not None:
        plan = replace(plan, count=min(plan.count, contract["max_windows"]))
    size = _batch_size(cfg, len(entry["output_channels"]), [m["identifier"] for m in contract["metrics"]])
    return {"schema": RUN_SCHEMA, "contract": contract, "state": "pending",
            "channels": entry["output_channels"], "recording_metadata": metadata_link,
            "windows": plan.as_dict(include_ranges=False), "dtypes": contract["output_dtypes"].copy(), "sessions": [],
            "batches": [{"start": start, "stop": min(start + size, plan.count), "state": "pending"}
                        for start in range(0, plan.count, size)]}


def _events_for_batch(journal, batch, identifiers):
    try:
        return [journal.latest[(window, metric)] for window in range(batch["start"], batch["stop"])
                for metric in identifiers]
    except KeyError as exc:
        raise WorkflowError("cannot commit a batch with unattempted metric/windows") from exc


def _validate_committed(path, manifest):
    """Check declarations on resume without scanning every compressed payload."""
    from pyspi.storage import open_results

    identifiers = [m["identifier"] for m in manifest["contract"]["metrics"]]
    next_window = 0
    for batch in manifest["batches"]:
        if batch["start"] != next_window or not batch["start"] < batch["stop"] <= manifest["windows"]["count"]:
            raise WorkflowError("invalid declared window coverage in run batches")
        next_window = batch["stop"]
        if batch["state"] != "complete":
            continue
        root = child_path(path, batch["path"])
        try:
            store = open_results(root / "results.zarr")
            if store.metrics != identifiers or store.recording.get("channels") != manifest["channels"]:
                raise WorkflowError("committed batch metric/channel order differs")
            for metric in identifiers:
                meta = store.info(metric)
                if (meta["state"] != "complete" or meta["n"] != len(manifest["channels"]) or
                    meta["windows"] != batch["stop"] - batch["start"] or meta["single_matrix"]):
                    raise WorkflowError("committed batch declaration is incomplete or has different axes")
            events = read_json(root / "reports.json")
            expected = [(w, m) for w in range(batch["start"], batch["stop"]) for m in identifiers]
            if [(e["window"], e["report"]["identifier"]) for e in events] != expected or any(
                    e["report"]["status"] not in ("success", "failed") for e in events):
                raise WorkflowError("committed diagnostics disagree with batch axes")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WorkflowError(f"cannot resume unreadable/inconsistent committed batch {root}: {exc}") from exc
    if next_window != manifest["windows"]["count"]:
        raise WorkflowError("incomplete declared window coverage in run batches")


def _commit_batch(path, manifest, batch, buffers, journal, metrics, session):
    from pyspi.storage import write_results
    from pyspi.storage._integrity import sync_directory

    started = perf_counter()
    base = path / "batches" / f"{batch['start']:08d}"
    generation = base / f"attempt-{uuid4().hex}"
    generation.mkdir()
    identifiers = [m.identifier for m in metrics]
    events = _events_for_batch(journal, batch, identifiers)
    successful = {e["report"]["identifier"] for e in events if e["report"]["status"] == "success"}
    n, count = len(manifest["channels"]), batch["stop"] - batch["start"]
    for identifier in identifiers:
        if identifier not in buffers:
            buffers[identifier] = np.full((n, n, count), np.nan,
                                         dtype=manifest["dtypes"].get(identifier, "float64"))
        dtype = buffers[identifier].dtype.str
        if identifier in successful and identifier in manifest["dtypes"] and manifest["dtypes"][identifier] != dtype:
            raise WorkflowError(f"{identifier}: output dtype changed between batches")
        if identifier in successful:
            manifest["dtypes"][identifier] = dtype
    write_results(generation / "results.zarr", {m: buffers[m] for m in identifiers},
                  labels={m["identifier"]: m["labels"] for m in manifest["contract"]["metrics"]},
                  attrs={m.identifier: {"module": m.module, "class_name": m.class_name,
                                        "parameters": m.parameters} for m in metrics},
                  recording={"channels": manifest["channels"], "start_window": batch["start"],
                             "stop_window": batch["stop"], "run": "../../../../run.json"})
    journal.sync()
    atomic_json(generation / "reports.json", events)
    sync_directory(base)
    committed = {**batch, "state": "complete", "path": str(generation.relative_to(path))}
    candidate = {**manifest, "batches": [committed if b is batch else b for b in manifest["batches"]]}
    atomic_json(path / "run.json", candidate)
    batch.update(committed)
    session["output_seconds"] += perf_counter() - started
    # Old generations are no longer referenced. Only our validated run's batch
    # staging directories are removed; existing patient stores are never here.
    for old in base.glob("attempt-*"):
        if old != generation and old.is_dir() and not old.is_symlink():
            shutil.rmtree(old)


def _store_values(buffers, report, values, valid, n, count, slot, dtypes):
    if values is None:
        return
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")) or values.shape != (len(valid), len(valid)):
        raise WorkflowError("Calculator produced unsupported output dtype/axes")
    values, conversion = convert_output(values, dtypes[report.identifier], report.identifier)
    if conversion is not None:
        conversion["positions"] = [[int(valid[row]), int(valid[column])]
                                   for row, column in conversion["positions"]]
        conversion["position_axes"] = "zero_based_output_row_column"
    if report.identifier not in buffers:
        buffers[report.identifier] = np.full((n, n, count), np.nan, dtype=values.dtype)
    if buffers[report.identifier].dtype != values.dtype:
        raise WorkflowError("Calculator dtype changed within a batch")
    target = buffers[report.identifier][:, :, slot]
    target[:] = np.nan
    target[np.ix_(valid, valid)] = values
    return conversion


def _execute_batch(path, manifest, batch, prepared, calc, metrics, retry_failures, session, pool=None):
    from pyspi.calculation import MetricReport, _capture_metric_warnings, _describe_error
    from pyspi.data import InvalidChannelsError
    from pyspi.storage import open_results

    identifiers, full_spis = [m.identifier for m in metrics], dict(calc.spis)
    base = path / "batches" / f"{batch['start']:08d}"
    journal = Journal(base / "attempts.jsonl")
    buffers = {}
    committed = batch["state"] == "complete"
    n, count = len(manifest["channels"]), batch["stop"] - batch["start"]
    try:
        if committed:
            old = child_path(path, batch["path"])
            events = read_json(old / "reports.json")
            for event in events:
                journal.latest.setdefault((event["window"], event["report"]["identifier"]), event)
            failures = {(e["window"], e["report"]["identifier"]) for e in events if e["report"]["status"] == "failed"}
            if not retry_failures or not failures:
                return
            # Preserve every nonfailed value bit-for-bit while replacing only
            # explicitly retried failures. Never accept unreadable source data.
            store = open_results(old / "results.zarr")
            for identifier in identifiers:
                values, good = store.load(identifier, return_availability=True)
                if not good.all():
                    raise WorkflowError("cannot retry failures in a batch with unreadable stored results")
                known = manifest["dtypes"].get(identifier)
                if known and values.dtype != np.dtype(known):
                    if any(e["report"]["status"] == "success" and e["report"]["identifier"] == identifier for e in events):
                        raise WorkflowError("committed successful values disagree with run dtype")
                    values = np.full(values.shape, np.nan, dtype=known)
                buffers[identifier] = values
        else:
            failures = set()
        if pool is not None:
            jobs = []
            for window in range(batch["start"], batch["stop"]):
                active = ([m for m in identifiers if (window, m) in failures] if committed else
                          [m for m in identifiers if retry_failures or
                           journal.latest.get((window, m), {}).get("report", {}).get("status") != "failed"])
                if active:
                    jobs.append((window, active))
            before = perf_counter()
            try:
                for result in pool.map(jobs):
                    session["window_preprocessing_seconds"] += result.preprocessing_seconds
                    session.setdefault("worker_peak_rss_mib", {})[str(result.pid)] = result.peak_rss_mib
                    session.setdefault("worker_jvm_started", {})[str(result.pid)] = result.jvm_started
                    output_started = perf_counter()
                    for report, values in result.records:
                        session["metric_elapsed_seconds"] += report.elapsed_seconds
                        conversion = _store_values(buffers, report, values, result.valid, n, count,
                                                   result.window-batch["start"], manifest["dtypes"])
                        journal.record(result.window, report, stage=result.stage, window_warnings=result.warnings,
                                       output_conversion=conversion)
                    journal.sync()
                    session["output_seconds"] += perf_counter()-output_started
            finally:
                # Includes worker setup, preprocessing and IPC. Summed metric
                # times describe work and can exceed this parallel wall time.
                session["calculation_wall_seconds"] += perf_counter()-before
            _commit_batch(path, manifest, batch, buffers, journal, metrics, session)
            return
        for window in range(batch["start"], batch["stop"]):
            if committed:
                active = [m for m in identifiers if (window, m) in failures]
            else:
                active = [m for m in identifiers if retry_failures or
                          journal.latest.get((window, m), {}).get("report", {}).get("status") != "failed"]
            if not active:
                continue
            # Public SPI mapping is temporarily filtered; load_dataset creates a
            # fresh Data/cache and correctly shaped table for this window only.
            calc.spis.clear()
            calc.spis.update((m, full_spis[m]) for m in active)
            start = window * manifest["windows"]["step_samples"]
            stop = start + manifest["windows"]["length_samples"]
            before = perf_counter()
            from ._reference import available_indices
            valid = available_indices(prepared.details, n)
            try:
                with _capture_metric_warnings() as window_warnings, redirect_stdout(sys.stderr):
                    while valid.size >= 2:
                        try:
                            from ._bands import window_band_views
                            calc.load_dataset(prepared.data[valid, start:stop].copy(),
                                filtered_bands=window_band_views(prepared.bands, valid, start, stop) or None,
                                sampling_rate_hz=float(manifest["contract"]["sampling_rate_hz"]))
                            break
                        except InvalidChannelsError as exc:
                            excluded = [manifest["channels"][i] for i in valid[exc.processes]]
                            warnings.warn(f"Window {window}: skipping unusable channels {excluded}: {exc}",
                                          UserWarning)
                            valid = np.delete(valid, exc.processes)
                    if valid.size < 2:
                        raise ValueError("Fewer than two usable channels remain in this window.")
            except Exception as exc:
                error = _describe_error(exc)
                for identifier in active:
                    journal.record(window, MetricReport(identifier, "failed", 0., tuple(window_warnings), error),
                                   stage="window_preprocessing")
                journal.sync()
                continue
            finally:
                session["window_preprocessing_seconds"] += perf_counter() - before

            def complete(report, values):
                before = perf_counter()
                session["metric_elapsed_seconds"] += report.elapsed_seconds
                conversion = _store_values(buffers, report, values, valid, n, count,
                                           window-batch["start"], manifest["dtypes"])
                journal.record(window, report, window_warnings=window_warnings, output_conversion=conversion)
                session["output_seconds"] += perf_counter() - before

            before = perf_counter()
            try:
                with redirect_stdout(sys.stderr):
                    calc.compute(on_metric_complete=complete)
            finally:
                session["calculation_wall_seconds"] += perf_counter() - before
            journal.sync()
        _commit_batch(path, manifest, batch, buffers, journal, metrics, session)
    finally:
        calc.spis.clear()
        calc.spis.update(full_spis)
        journal.close()


def run(config, *, resume=False, retry_failures=False, max_windows=None):
    """Execute with configured process/thread/cache limits, preserving caller limits."""
    from ._config import Configuration, load_config
    from ._parallel import numerical_threads
    cfg = config if isinstance(config, Configuration) else load_config(config)
    with numerical_threads(cfg.execution["threads_per_worker"]):
        return _run(cfg, resume=resume, retry_failures=retry_failures, max_windows=max_windows)


def _run(config, *, resume=False, retry_failures=False, max_windows=None):
    """Return one run-folder path per recording.

    Resume reuses committed batches and recomputes unfinished batches. Durable
    failures are skipped unless retry_failures=True, which requires resume=True.
    No RNG seed/reset or estimator parameter rewriting occurs here.
    """
    from pyspi.calculation import _capture_metric_warnings, _describe_error
    try:
        import pyspi.storage  # noqa: F401; validate optional execution dependencies
    except ImportError as exc:
        raise WorkflowError("EEG execution requires pyspi[storage]; install the storage extra") from exc

    if retry_failures and not resume:
        raise WorkflowError("retry_failures requires explicit resume")
    if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
        raise WorkflowError("max_windows must be a positive integer")
    invocation_started = perf_counter()
    report = preflight(config)
    cfg = report.configuration
    if cfg.output_directory is None:
        raise WorkflowError("output_directory is required for execution")
    environment, output = _environment(), []
    with tempfile.TemporaryDirectory(prefix="pyspi-workflow-") as temp, _class_label_state(report.metrics):
        calc = _calculator(report.metrics, cfg, Path(temp))
        setup_seconds = perf_counter() - invocation_started
        destinations = [(r.patient_id, cfg.run_name or r.recording_id) for r in cfg.recordings]
        if len(set(destinations)) != len(destinations):
            raise WorkflowError("run_name maps multiple recordings to the same patient output; run them separately")
        if resume and not any((cfg.output_directory / patient / "calculations" / name / "run.json").is_file()
                              for patient, name in destinations):
            raise WorkflowError("no run to resume among the configured recordings")
        for recording, entry in zip(cfg.recordings, report.recordings):
            recording_started = perf_counter()
            patient = cfg.output_directory / recording.patient_id
            path = patient / "calculations" / (cfg.run_name or recording.recording_id)
            if not path.resolve().is_relative_to(cfg.output_directory.resolve()):
                raise WorkflowError("output path escapes output_directory through a symlink")
            if any((parent / "zarr.json").exists() or (parent / ".zgroup").exists() for parent in (path, *path.parents)):
                raise WorkflowError("workflow output must be outside existing Zarr stores")
            contract = _contract(cfg, recording, entry, report.metrics, environment, max_windows)
            with _run_lock(path):
                if (path / "run.json").exists():
                    manifest = read_json(path / "run.json")
                    if not resume:
                        raise WorkflowError(f"output already exists; use --resume or a different run_name: {path}")
                    if (manifest.get("schema") != RUN_SCHEMA or
                        _compatibility_fields(manifest.get("contract", {})) != _compatibility_fields(contract)):
                        raise WorkflowError(f"incompatible resume configuration/source/environment: {path}")
                    _validate_committed(path, manifest)
                    for previous in manifest["sessions"]:
                        if previous["status"] == "running":
                            # Acquiring the writer lock proves this session no
                            # longer has a writer. A hard kill has no final wall
                            # time; do not invent one from partially saved totals.
                            previous.update(status="interrupted", elapsed_seconds=None,
                                            ended_without_final_report=True)
                else:
                    # A cohort interruption may occur before later recordings
                    # even have a run manifest. Resume starts those normally.
                    metadata = patient / "metadata" / f"{cfg.run_name or recording.recording_id}.json"
                    identity = {"schema": "pyspi-recording/1", "patient_id": recording.patient_id,
                                "recording_id": recording.recording_id, "source": contract["source"],
                                "channels": entry["channels"],
                                "selection_metadata": str(cfg.metadata) if cfg.metadata else None}
                    if metadata.exists() and read_json(metadata) != identity:
                        raise WorkflowError(f"recording metadata already differs: {metadata}")
                    atomic_json(metadata, identity)
                    manifest = _new_manifest(contract, entry, cfg, str(Path("../../metadata") / metadata.name))
                    atomic_json(path / "run.json", manifest)
                session = {"preparation_seconds": 0., "window_preprocessing_seconds": 0.,
                           "calculation_wall_seconds": 0., "metric_elapsed_seconds": 0.,
                           "output_seconds": 0., "elapsed_seconds": 0.,
                           "resume": resume, "retry_failures": retry_failures, "status": "running"}
                session["invocation_setup_seconds"] = setup_seconds
                session["execution"] = cfg.execution.copy()
                manifest["sessions"].append(session)
                manifest["state"] = "running"
                atomic_json(path / "run.json", manifest)
                try:
                    needed = [b for b in manifest["batches"] if b["state"] != "complete"]
                    if retry_failures:
                        needed = manifest["batches"]
                    if needed:
                        before = perf_counter()
                        try:
                            with _capture_metric_warnings() as preparation_warnings:
                                from ._bands import required_bands
                                prepared = prepare_recording(recording, cfg, bands=required_bands(report.metrics, cfg))
                        finally:
                            session["preparation_seconds"] = perf_counter() - before
                        manifest["preparation"] = {**prepared.details,
                            "warnings": [asdict(w) for w in preparation_warnings]}
                        if _source_identity(recording.path) != contract["source"]:
                            raise WorkflowError("EDF source changed during preparation")
                        actual = plan_windows(prepared.data.shape[1], cfg.sampling_rate_hz,
                                              cfg.window_seconds, cfg.step_seconds,
                                              recording=f"{recording.patient_id}/{recording.recording_id}")
                        if max_windows is not None:
                            actual = replace(actual, count=min(actual.count, max_windows))
                        if actual.as_dict(include_ranges=False) != manifest["windows"] or prepared.channel_names != manifest["channels"]:
                            raise WorkflowError("prepared channel/window axes disagree with the declared run")
                        atomic_json(path / "run.json", manifest)
                        from ._parallel import WindowPool
                        pool = None
                        interrupted = True
                        try:
                            if cfg.execution["workers"] > 1:
                                pool = WindowPool(prepared, report.metrics, cfg, manifest["windows"])
                                session["parallel_input_setup_seconds"] = pool.setup_seconds
                            for batch in needed:
                                _execute_batch(path, manifest, batch, prepared, calc, report.metrics,
                                               retry_failures, session, pool=pool)
                            interrupted = False
                        finally:
                            if pool is not None:
                                pool.close(interrupted=interrupted)
                        del prepared
                    manifest["state"] = "complete"
                    session["status"] = "complete"
                except BaseException as exc:
                    session["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
                    session["error"] = asdict(_describe_error(exc))
                    manifest["state"] = session["status"]
                    raise
                finally:
                    from ._parallel import peak_rss_mib
                    session["parent_peak_rss_mib"] = peak_rss_mib()
                    session["elapsed_seconds"] = perf_counter() - recording_started
                    atomic_json(path / "run.json", manifest)
            output.append(path)
        summary = {"schema": "pyspi-eeg-invocation/1", "configuration": str(cfg.source),
                   "runs": [str(p.relative_to(cfg.output_directory)) for p in output],
                   "setup_seconds": setup_seconds, "elapsed_seconds": perf_counter() - invocation_started,
                   "resume": resume, "retry_failures": retry_failures}
        atomic_json(cfg.output_directory / f"execution-{uuid4().hex}.json", summary)
    return output
