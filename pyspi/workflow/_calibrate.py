"""Measure per-window cost across channel counts on one prepared recording.

One recording is prepared exactly as a run prepares it; its first M channels
are timed through the configured Calculator for each ladder count M, up to the
recording's full channel count, so estimates interpolate rather than
extrapolate. A batch commit at the full count measures output bytes and write
cost. Synthetic data is a fallback for configurations without usable EDFs.
"""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import numpy as np

from ._config import Configuration, WorkflowError, load_config
from ._estimate import CALIBRATION_SCHEMA, sidecar_path
from ._output import atomic_json
from ._windows import plan_windows

LADDER = (2, 4, 8, 16, 32, 64, 128)
DEFAULT_BUDGET_SECONDS = 600.


def default_ladder(channels):
    if channels < 2:
        raise WorkflowError("calibration needs a recording with at least two output channels")
    return [count for count in LADDER if count < channels] + [channels]


def choose_recording(report):
    """Index with the most usable contacts; other computation counts interpolate."""
    if not report.recordings:
        raise WorkflowError("no recordings to calibrate")
    return max(range(len(report.recordings)), key=lambda i: (
        sum(report.recordings[i].get("reference_available", [True] * len(report.recordings[i]["output_channels"]))), -i))


def pick_windows(count, k):
    """``k`` window indices spread over the middle half of the recording."""
    if count < 1:
        raise WorkflowError("the calibration recording has no complete windows")
    k = min(k, count)
    low, high = count // 4, max(count // 4, (3 * count) // 4 - 1)
    return sorted({int(round(low + (high - low) * i / max(k - 1, 1))) for i in range(k)})


def synthetic_block(channels, samples, seed=0):
    """AR(1) stand-in for signal timing only; not for scientific validity."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((channels, samples))
    block = np.zeros((channels, samples))
    for t in range(1, samples):
        block[:, t] = .6 * block[:, t - 1] + .2 * np.roll(block[:, t - 1], 1) + noise[:, t]
    return block


def _time_window(calc, block, names, start, stop, *, source_indices=None):
    from pyspi.calculation import _capture_metric_warnings
    from pyspi.data import InvalidChannelsError

    valid = np.arange(len(names))
    records = []
    started = perf_counter()
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet), \
            _capture_metric_warnings() as emitted:
        while valid.size >= 2:
            try:
                indices = valid if source_indices is None else source_indices[valid]
                calc.load_dataset(block[indices, start:stop].copy())
                break
            except InvalidChannelsError as exc:
                valid = np.delete(valid, exc.processes)
        if valid.size < 2:
            raise WorkflowError("fewer than two usable channels remain in a calibration window")
        preprocessing = perf_counter() - started
        before = perf_counter()
        calc.compute(on_metric_complete=lambda report, values: records.append((report, values)))
        compute = perf_counter() - before
    return {"preprocessing_seconds": preprocessing, "compute_seconds": compute,
            "records": records, "valid": valid, "warnings": len(emitted)}


def _time_commit(directory, buffers, events, metrics, names, count):
    """Seconds for one durable batch commit of ``count`` windows, mirroring the runner."""
    from pyspi.storage import write_results
    from pyspi.storage._integrity import sync_directory

    base = Path(directory) / f"commit-{count}"
    base.mkdir(parents=True)
    started = perf_counter()
    write_results(base / "results.zarr", {m.identifier: buffers[m.identifier][:, :, :count] for m in metrics},
                  labels={m.identifier: list(m.labels) for m in metrics},
                  attrs={m.identifier: {"module": m.module, "class_name": m.class_name,
                                        "parameters": m.parameters} for m in metrics},
                  recording={"channels": names, "start_window": 0, "stop_window": count})
    atomic_json(base / "reports.json", [e for e in events if e["window"] < count])
    sync_directory(base)
    return perf_counter() - started, base


def _tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _measure_output(directory, buffers, events, metrics, names, k):
    n = len(names)
    raw_per_window = sum(n * n * buffers[m.identifier].itemsize for m in metrics)
    seconds_one, base_one = _time_commit(directory, buffers, events, metrics, names, 1)
    if k > 1:
        seconds_k, base = _time_commit(directory, buffers, events, metrics, names, k)
        per_byte = max(0., (seconds_k - seconds_one) / (raw_per_window * (k - 1)))
        per_batch = max(0., seconds_one - per_byte * raw_per_window)
    else:
        seconds_k, base, per_byte, per_batch = seconds_one, base_one, 0., seconds_one
    stored = {m.identifier: _tree_bytes(base / "results.zarr" / m.identifier) / (k * n * n * buffers[m.identifier].itemsize)
              for m in metrics}
    reports = (base / "reports.json").stat().st_size / (k * len(metrics))
    return {"channels": n, "windows": k, "seconds_per_batch": per_batch, "seconds_per_raw_byte": per_byte,
            "commit_seconds": {"1": seconds_one, str(k): seconds_k}, "raw_bytes_per_window": raw_per_window,
            "stored_ratio": stored, "report_bytes_per_event": reports,
            "limits": "One commit each of one and k windows to a temporary directory on this host's "
                      "filesystem; fsync cost and compression ratios depend on the destination and the data."}


def _predicted_point_seconds(points, count):
    """Expected measurement time for the next ladder count from the last two points."""
    (c0, t0), (c1, t1) = [(p["channels"], p["window_seconds"]) for p in points[-2:]]
    exponent = math.log(t1 / t0) / math.log(c1 / c0) if t0 > 0 and t1 > 0 else 2.
    exponent = min(max(exponent, 1.), 3.)
    return t1 * (count / c1) ** exponent * points[-1]["windows"]


def calibrate(config, *, out=None, channels=None, windows_per_point=2, synthetic=False,
              budget_seconds=DEFAULT_BUDGET_SECONDS):
    """Measure the configuration's metric set on one recording; return and save the calibration.

    The ladder stops early, after at least two counts, when the next count is
    predicted to exceed the remaining budget; estimates then extrapolate from
    the measured counts and say so. Costly metric sets (Java information
    estimators, convergent cross mapping, wavelets) need this to finish at all.
    """
    from pyspi.calculation import _capture_metric_warnings
    from ._parallel import numerical_threads
    from ._prepare import prepare_recording
    from ._preflight import preflight
    from ._run import _calculator, _class_label_state, _environment, _store_values

    cfg = config if isinstance(config, Configuration) else load_config(config)
    if type(windows_per_point) is not int or windows_per_point < 1:
        raise WorkflowError("windows_per_point must be a positive integer")
    if not (isinstance(budget_seconds, (int, float)) and budget_seconds > 0):
        raise WorkflowError("budget_seconds must be positive")
    out = sidecar_path(cfg) if out is None else Path(out)
    started = perf_counter()
    report = preflight(cfg)
    index = choose_recording(report)
    entry, recording = report.recordings[index], cfg.recordings[index]
    all_names = entry["output_channels"]
    available = np.flatnonzero(entry.get("reference_available", [True] * len(all_names)))
    names = [all_names[i] for i in available]
    total = len(names)
    ladder = default_ladder(total) if channels is None else sorted({int(c) for c in channels})
    if not ladder or ladder[0] < 2 or ladder[-1] > total:
        raise WorkflowError(f"calibration channel counts must lie within 2..{total} for "
                            f"{entry['patient_id']}/{entry['recording_id']}")
    window = int(cfg.window_seconds * cfg.sampling_rate_hz)
    with numerical_threads(cfg.execution["threads_per_worker"]), \
            tempfile.TemporaryDirectory(prefix="pyspi-calibrate-") as temp, _class_label_state(report.metrics):
        calc = _calculator(report.metrics, cfg, Path(temp))
        setup_seconds = perf_counter() - started   # preflight, metric construction and Calculator, as a run pays
        if synthetic:
            step = int(cfg.step_seconds * cfg.sampling_rate_hz)
            block = synthetic_block(total, (windows_per_point - 1) * step + window)
            plan = plan_windows(block.shape[1], cfg.sampling_rate_hz, cfg.window_seconds, cfg.step_seconds)
            preparation = None
            source = {"kind": "synthetic", "patient_id": None, "recording_id": None, "path": None}
        else:
            before = perf_counter()
            with _capture_metric_warnings():
                prepared = prepare_recording(recording, cfg)
            preparation = {"seconds": perf_counter() - before, "channels": len(entry["channels"]),
                           "native_samples": sum(c["native_samples"] for c in entry["channels"]),
                           "edf_bytes": recording.path.stat().st_size, "details": prepared.details}
            if prepared.channel_names != all_names:
                raise WorkflowError("prepared channels disagree with the preflight channel order")
            block = prepared.data
            plan = plan_windows(block.shape[1], cfg.sampling_rate_hz, cfg.window_seconds, cfg.step_seconds,
                                recording=f"{entry['patient_id']}/{entry['recording_id']}")
            source = {"kind": "edf", "patient_id": entry["patient_id"], "recording_id": entry["recording_id"],
                      "path": entry["path"]}
        windows = pick_windows(plan.count, windows_per_point)
        source.update(output_channels=len(all_names), calculation_channels=total,
                      windows_used=windows, windows_available=plan.count)
        points, buffers, events, dtypes = [], {}, [], {}
        budget = {"seconds": float(budget_seconds), "stopped_before_channels": None, "predicted_seconds": None}
        measuring_started = perf_counter()
        for count in ladder:
            if len(points) >= 2:
                predicted = _predicted_point_seconds(points, count)
                if perf_counter() - measuring_started + predicted > budget_seconds:
                    budget.update(stopped_before_channels=count, predicted_seconds=predicted)
                    print(f"calibration budget reached; skipping {count}+ channels (predicted {predicted:.0f} s)",
                          file=sys.stderr)
                    break
            selected = np.arange(count) if synthetic else available[:count]
            timings = [_time_window(calc, block, names[:count], w * plan.step_samples,
                                    w * plan.step_samples + plan.length_samples, source_indices=selected)
                       for w in windows]
            metrics = {}
            for m in report.metrics:
                seconds = [next(r.elapsed_seconds for r, _ in t["records"] if r.identifier == m.identifier)
                           for t in timings]
                statuses = [next(r.status for r, _ in t["records"] if r.identifier == m.identifier) for t in timings]
                metrics[m.identifier] = {"seconds": float(np.mean(seconds)),
                                         "status": "success" if all(s == "success" for s in statuses) else "failed"}
            preprocessing = float(np.mean([t["preprocessing_seconds"] for t in timings]))
            compute = float(np.mean([t["compute_seconds"] for t in timings]))
            totals = [t["preprocessing_seconds"] + t["compute_seconds"] for t in timings]
            # The slowest timed window stands for the recording: windows and
            # patients vary, and a late run costs more than a slack estimate.
            points.append({"channels": count, "windows": len(windows), "preprocessing_seconds": preprocessing,
                           "compute_seconds": compute, "window_seconds": float(max(totals)),
                           "mean_window_seconds": float(np.mean(totals)),
                           "metric_elapsed_sum_seconds": sum(v["seconds"] for v in metrics.values()),
                           "warnings": sum(t["warnings"] for t in timings), "metrics": metrics})
            print(f"calibrated {count} channels: {max(totals):.2f} s per window (slowest of {len(totals)})",
                  file=sys.stderr)
            # The top measured count supplies the output measurement; keep its matrices.
            buffers, events = {}, []
            for slot, timing in enumerate(timings):
                for r, values in timing["records"]:
                    _store_values(buffers, r, values, timing["valid"], count, len(windows), slot)
                    events.append({"window": slot, "stage": "calculation", "report": r.as_dict()})
            for m in report.metrics:
                if m.identifier not in buffers:
                    buffers[m.identifier] = np.full((count, count, len(windows)), np.nan)
                dtypes[m.identifier] = buffers[m.identifier].dtype.str
        measured = points[-1]["channels"]
        output = _measure_output(Path(temp) / "output", buffers, events, report.metrics, names[:measured], len(windows))
    result = {"schema": CALIBRATION_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": str(cfg.source),
              "config_sha256": hashlib.sha256(cfg.source.read_bytes()).hexdigest(),
              "metric_config_sha256": hashlib.sha256(cfg.metric_config.read_bytes()).hexdigest(),
              "environment": _environment(),
              "machine": {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
                          "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                          "load_average_before": list(os.getloadavg()) if hasattr(os, "getloadavg") else None},
              "execution": cfg.execution.copy(), "window_samples": window,
              "sampling_rate_hz": float(cfg.sampling_rate_hz), "detrend": cfg.detrend, "zscore": cfg.zscore,
              "metrics": [m.identifier for m in report.metrics], "dtypes": dtypes, "source": source,
              "calculator_setup_seconds": setup_seconds, "preparation": preparation,
              "ladder": ladder, "budget": budget,
              "points": points, "output": output, "elapsed_seconds": perf_counter() - started,
              "limits": "Single-recording, single-host measurement with one numerical thread per worker as "
                        "configured. Data-dependent estimators vary by window and patient; spectral cache "
                        "effects beyond the calibrated channel count are not modelled."}
    atomic_json(out, result)
    return result
