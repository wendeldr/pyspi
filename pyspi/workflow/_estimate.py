"""Rough runtime and output-size estimates from a measured channel-count calibration.

Estimates are interpolated between measured channel counts of one calibration
recording; above the largest measured count they are extrapolated with the
local exponent and flagged. Data-dependent estimators, host contention and the
spectral cache policy all move real times; treat results as planning figures.
"""

import math
from pathlib import Path
from statistics import median

import numpy as np

from ._config import Configuration, WorkflowError, load_config
from ._output import read_json

CALIBRATION_SCHEMA = "pyspi-workflow-calibration/1"
PARALLEL_EFFICIENCY = .9          # measured .97 with two workers; keep a margin for more
EXTRAPOLATION_EXPONENT = (0., 3.)  # clamp for the local channel exponent above the ladder
BREAKDOWN_TOLERANCE = .25          # per-metric sum versus pooled wall disagreement that is reported
# Estimates should err long and large: a run that finishes early costs nothing,
# one that overruns costs a schedule. Calibration measures a clean single
# process; real runs add journals, callbacks, memory pressure and other jobs.
HEADROOM = {"prepare": 1.25, "compute": 1.15, "output": 1.4, "stored": 1.1}


def sidecar_path(cfg):
    return cfg.source.with_name(cfg.source.name + ".calibration.json")


def load_calibration(path):
    data = read_json(path)
    if not isinstance(data, dict) or data.get("schema") != CALIBRATION_SCHEMA or not data.get("points"):
        raise WorkflowError(f"not a {CALIBRATION_SCHEMA} calibration: {path}")
    return data


def curve(points, n):
    """``(seconds, mode)`` at ``n`` channels from measured ``(channels, seconds)`` points.

    Exact at a measured count, log-log linear between neighbours, flat below
    the smallest count, and a clamped local power law above the largest.
    """
    points = sorted((int(c), float(t)) for c, t in points)
    if not points:
        raise ValueError("at least one measured point is required")
    if n < 1:
        raise ValueError("channel count must be positive")
    for channels, seconds in points:
        if channels == n:
            return seconds, "measured"
    if n < points[0][0]:
        return points[0][1], "clamped"
    if n > points[-1][0]:
        if len(points) > 1:
            (c0, t0), (c1, t1) = points[-2:]
            exponent = math.log(t1 / t0) / math.log(c1 / c0) if t0 > 0 and t1 > 0 else 2.
        else:
            (c1, t1), exponent = points[-1], 2.
        exponent = min(max(exponent, EXTRAPOLATION_EXPONENT[0]), EXTRAPOLATION_EXPONENT[1])
        return t1 * (n / c1) ** exponent, "extrapolated"
    (c0, t0), (c1, t1) = next((a, b) for a, b in zip(points, points[1:]) if a[0] < n < b[0])
    fraction = math.log(n / c0) / math.log(c1 / c0)
    if t0 > 0 and t1 > 0:
        return math.exp(math.log(t0) + fraction * (math.log(t1) - math.log(t0))), "interpolated"
    return t0 + (n - c0) / (c1 - c0) * (t1 - t0), "interpolated"


def window_seconds(calibration, n):
    """Pooled per-window seconds at ``n`` channels, with the per-metric breakdown."""
    points = calibration["points"]
    seconds, mode = curve([(p["channels"], p["window_seconds"]) for p in points], n)
    preprocessing, _ = curve([(p["channels"], p["preprocessing_seconds"]) for p in points], n)
    metrics = {}
    for identifier in calibration["metrics"]:
        metrics[identifier], _ = curve([(p["channels"], p["metrics"][identifier]["seconds"]) for p in points], n)
    return {"seconds": seconds, "mode": mode, "preprocessing_seconds": preprocessing,
            "metric_sum_seconds": preprocessing + sum(metrics.values()), "metrics": metrics}


def calibration_issues(calibration, cfg, report):
    """``(stale, notes)``: reasons the calibration cannot describe this config, and softer caveats."""
    from ._run import _environment

    stale, notes = [], []
    identifiers = [m.identifier for m in report.metrics]
    if list(calibration["metrics"]) != identifiers:
        stale.append("metric set differs from the calibration (shared caches make per-metric times non-additive)")
    from ._precision import resolve_dtypes
    if any(calibration.get("dtypes", {}).get(name) != dtype
           for name, dtype in resolve_dtypes(cfg, identifiers).items()
           if name in calibration["metrics"]):
        stale.append("output precision differs from the calibration")
    window = int(cfg.window_seconds * cfg.sampling_rate_hz)
    if calibration["window_samples"] != window:
        stale.append(f"window length differs ({calibration['window_samples']} calibrated vs {window} samples)")
    for key, value in (("detrend", cfg.detrend), ("zscore", cfg.zscore)):
        if calibration[key] != value:
            stale.append(f"{key} differs from the calibration")
    preparation = calibration.get("preparation") or {}
    reference = preparation.get("details", {}).get("reference")
    if reference is not None and reference != cfg.reference:
        stale.append("reference differs from the calibration")
    if calibration["execution"]["threads_per_worker"] != cfg.execution["threads_per_worker"]:
        stale.append("threads_per_worker differs from the calibration")
    if calibration["execution"]["spectral_cache_mib"] != cfg.execution["spectral_cache_mib"]:
        notes.append("spectral_cache_mib differs from the calibration; phase-family cost depends on it")
    if calibration["environment"]["pyspi_code_sha256"] != _environment()["pyspi_code_sha256"]:
        notes.append("pyspi source changed since the calibration")
    if calibration["source"]["kind"] != "edf":
        notes.append("calibration used synthetic data; data-dependent estimators (robust covariance, "
                     "graphical lasso) and preparation are not represented")
    return stale, notes


def _itemsize(calibration, identifier):
    return np.dtype(calibration.get("dtypes", {}).get(identifier, "<f8")).itemsize


def estimate_recording(entry, cfg, calibration, *, max_windows=None):
    from ._run import _batch_size, planned_windows

    n = len(entry["output_channels"])
    identifiers = calibration["metrics"]
    plan = planned_windows(entry, cfg)
    count = plan.count if max_windows is None else min(plan.count, max_windows)
    usable = sum(entry.get("reference_available", [True] * n))
    if usable < 2:
        raise WorkflowError("fewer than two channels have an available reference")
    per_window = window_seconds(calibration, usable)
    batch = _batch_size(cfg, n, identifiers)
    batches = math.ceil(count / batch) if count else 0
    workers = cfg.execution["workers"]
    efficiency = 1. if workers == 1 else PARALLEL_EFFICIENCY
    seconds = per_window["seconds"]
    # Workers idle while the parent commits each batch: on average half a window each.
    compute = count * seconds / (workers * efficiency) + batches * seconds * (workers - 1) / (2 * workers)
    compute *= HEADROOM["compute"]
    preparation = calibration["preparation"]
    prepare = None
    native_samples = sum(c["native_samples"] for c in entry["channels"])
    if preparation is not None and preparation["native_samples"]:
        prepare = preparation["seconds"] / preparation["native_samples"] * native_samples * HEADROOM["prepare"]
    output = calibration["output"]
    from ._precision import resolve_dtypes
    raw_per_window = sum(n * n * np.dtype(dt).itemsize for dt in resolve_dtypes(cfg, identifiers).values())
    raw_bytes = count * raw_per_window
    output_seconds = (batches * output["seconds_per_batch"] + raw_bytes * output["seconds_per_raw_byte"]) * HEADROOM["output"]
    stored_bytes = count * sum(output["stored_ratio"][m] * n * n * _itemsize(calibration, m) for m in identifiers)
    stored_bytes += count * len(identifiers) * output["report_bytes_per_event"]
    stored_bytes *= HEADROOM["stored"]
    prepared_samples = max(1, round(entry["duration_seconds"] * float(cfg.sampling_rate_hz)))
    band_details = (calibration.get("preparation") or {}).get("details", {}).get("continuous_bands", [])
    band_specs = calibration.get("continuous_bands_hz")
    extra_bands = (sum(tuple(band) != cfg.bandpass for band in band_specs) if band_specs is not None
                   else sum(not band["reused_workflow_bandpass"] for band in band_details))
    return {"patient_id": entry["patient_id"], "recording_id": entry["recording_id"],
            "channels": n, "calculation_channels": usable, "windows": count, "batch_windows": batch, "batches": batches,
            "window_seconds": seconds, "mode": per_window["mode"],
            "prepare_seconds": prepare, "compute_seconds": compute, "output_seconds": output_seconds,
            "total_seconds": (prepare or 0.) + compute + output_seconds,
            "raw_output_bytes": raw_bytes, "stored_bytes": stored_bytes,
            "memory": {"prepared_bytes": (len(entry["channels"]) + extra_bands*n) * prepared_samples * 8,
                       "band_file_bytes": extra_bands * n * prepared_samples * 8,
                       "batch_buffer_bytes": batch * raw_per_window},
            "breakdown_sum_seconds": per_window["metric_sum_seconds"]}


def estimate(config, calibration=None, *, max_windows=None):
    """Estimate every configured recording; ``calibration`` defaults to the config's sidecar.

    A missing or stale sidecar is measured first. An explicitly supplied stale
    calibration is used with its issues reported in ``warnings``.
    """
    from ._preflight import preflight

    cfg = config if isinstance(config, Configuration) else load_config(config)
    report = preflight(cfg)
    notes = []
    if calibration is None:
        path = sidecar_path(cfg)
        data = load_calibration(path) if path.is_file() else None
        stale = (calibration_issues(data, cfg, report)[0] if data is not None
                 else ["no calibration beside the configuration"])
        if stale:
            from ._calibrate import calibrate
            notes.append(f"measured a new calibration at {path}: " + "; ".join(stale))
            data = calibrate(cfg, out=path)
        calibration = data
    elif not isinstance(calibration, dict):
        path = Path(calibration)
        calibration = load_calibration(path)
    else:
        path = None
    stale, soft = calibration_issues(calibration, cfg, report)
    notes.extend(f"calibration does not describe this configuration: {issue}" for issue in stale)
    notes.extend(soft)
    recordings = [estimate_recording(entry, cfg, calibration, max_windows=max_windows)
                  for entry in report.recordings]
    top = max(p["channels"] for p in calibration["points"])
    above = [r for r in recordings if r["calculation_channels"] > top]
    if above:
        worst = max(r["calculation_channels"] for r in above)
        budget = calibration.get("budget") or {}
        reason = (f" (the calibration stopped at its {budget['seconds']:.0f} s budget before "
                  f"{budget['stopped_before_channels']} channels)" if budget.get("stopped_before_channels") else "")
        notes.append(f"{len(above)} recording(s) exceed the calibrated {top} channels (up to {worst}, "
                     f"{worst / top:.1f}x); their times are extrapolated and cache effects are not modelled{reason}")
    typical = int(median(r["calculation_channels"] for r in recordings)) if recordings else top
    breakdown = window_seconds(calibration, typical)
    if breakdown["seconds"] > 1 and (abs(breakdown["metric_sum_seconds"] - breakdown["seconds"])
                                     > BREAKDOWN_TOLERANCE * breakdown["seconds"]):
        notes.append(f"per-metric times sum to {breakdown['metric_sum_seconds']:.2f} s but the pooled window took "
                     f"{breakdown['seconds']:.2f} s at {typical} channels; the calibration may be noisy")
    drivers = sorted(breakdown["metrics"].items(), key=lambda item: -item[1])[:10]
    workers = cfg.execution["workers"]
    setup = calibration["calculator_setup_seconds"] * (2 if workers > 1 else 1)
    totals = {"recordings": len(recordings), "windows": sum(r["windows"] for r in recordings),
              "setup_seconds": setup,
              "prepare_seconds": (None if any(r["prepare_seconds"] is None for r in recordings)
                                  else sum(r["prepare_seconds"] for r in recordings)),
              "compute_seconds": sum(r["compute_seconds"] for r in recordings),
              "output_seconds": sum(r["output_seconds"] for r in recordings),
              "total_seconds": setup + sum(r["total_seconds"] for r in recordings),
              "stored_bytes": sum(r["stored_bytes"] for r in recordings),
              "raw_output_bytes": sum(r["raw_output_bytes"] for r in recordings)}
    if recordings:  # for back-calculating cohorts of unknown size
        totals["mean_seconds_per_recording"] = sum(r["total_seconds"] for r in recordings) / len(recordings)
        totals["mean_stored_bytes_per_recording"] = totals["stored_bytes"] / len(recordings)
    return {"schema": "pyspi-workflow-estimate/1", "configuration": str(cfg.source),
            "calibration": None if path is None else str(path),
            "calibration_source": calibration["source"],
            "calibrated_channels": [p["channels"] for p in calibration["points"]],
            "execution": cfg.execution.copy(), "parallel_efficiency": PARALLEL_EFFICIENCY, "headroom": dict(HEADROOM),
            "max_windows": max_windows, "recordings": recordings, "totals": totals,
            "drivers": {"channels": typical, "window_seconds": breakdown["seconds"],
                        "metrics": [{"identifier": k, "seconds": v} for k, v in drivers]},
            "warnings": notes,
            "limits": "Interpolated from one calibration recording on this host. Data-dependent estimators, "
                      "host load, cache limits and resumed runs change real durations; output sizes assume "
                      "the calibrated compression ratios."}


def _duration(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def _size(count):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if count < 1000 or unit == "TB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1000


def format_estimate(result):
    lines = []
    header = ("recording", "ch", "windows", "s/window", "prepare", "compute", "output", "stored", "total", "basis")
    rows = [(f"{r['patient_id']}/{r['recording_id']}", str(r["channels"]), str(r["windows"]),
             f"{r['window_seconds']:.2f}", _duration(r["prepare_seconds"]), _duration(r["compute_seconds"]),
             _duration(r["output_seconds"]), _size(r["stored_bytes"]), _duration(r["total_seconds"]), r["mode"])
            for r in result["recordings"]]
    totals = result["totals"]
    execution = result["execution"]
    if len(rows) != 1:
        rows.append(("total", "", str(totals["windows"]), "", _duration(totals["prepare_seconds"]),
                     _duration(totals["compute_seconds"]), _duration(totals["output_seconds"]),
                     _size(totals["stored_bytes"]), _duration(totals["total_seconds"]), ""))
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    for row in (header, *rows):
        lines.append("  ".join(cell.ljust(w) if i in (0, 9) else cell.rjust(w)
                               for i, (cell, w) in enumerate(zip(row, widths))).rstrip())
    lines.append("")
    lines.append(f"Total: {_duration(totals['total_seconds'])} for {totals['recordings']} recording(s), "
                 f"{totals['windows']} windows, {execution['workers']} worker(s); "
                 f"prepare {_duration(totals['prepare_seconds'])}, compute {_duration(totals['compute_seconds'])}, "
                 f"output {_duration(totals['output_seconds'])}, setup {_duration(totals['setup_seconds'])}")
    lines.append(f"Output: about {_size(totals['stored_bytes'])} on disk "
                 f"({_size(totals['raw_output_bytes'])} uncompressed float matrices)")
    if "mean_seconds_per_recording" in totals:
        lines.append(f"Per recording (mean): {_duration(totals['mean_seconds_per_recording'])}, "
                     f"{_size(totals['mean_stored_bytes_per_recording'])}")
    headroom = result["headroom"]
    lines.append("Figures err on the long side: slowest calibrated window, plus headroom of "
                 + ", ".join(f"{int(round((v - 1) * 100))}% {k}" for k, v in headroom.items()))
    drivers = result["drivers"]
    lines.append(f"Largest per-window costs at {drivers['channels']} channels "
                 f"({drivers['window_seconds']:.2f} s per window):")
    for item in drivers["metrics"]:
        lines.append(f"  {item['seconds']:8.3f} s  {item['identifier']}")
    source = result["calibration_source"]
    lines.append(f"Calibration: {source['kind']} data, {source.get('patient_id') or '-'}/"
                 f"{source.get('recording_id') or '-'}, channels "
                 f"{', '.join(str(c) for c in result['calibrated_channels'])}"
                 + (f" ({result['calibration']})" if result["calibration"] else ""))
    for note in result["warnings"]:
        lines.append(f"Warning: {note}")
    return "\n".join(lines)
