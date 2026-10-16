"""Bounded serial baseline. Run --prepare, then --case 50/118/249 separately.

Set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 before Python starts.
This harness observes existing implementations; it does not change estimators.
"""

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_info
import yaml

from pyspi.calculator import Calculator
from pyspi.storage import open_results, write_results
from pyspi.workflow import export_selection, load_config, open_run, run

runner = importlib.import_module("pyspi.workflow._run")
WINDOWS = 3
CASES = {50: "001", 118: "001", 249: "111"}
PROGRESS = sys.stdout


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def digest(array):
    return sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def sources_unchanged():
    snapshots = json.loads((BASE / "inputs.json").read_text())["source_snapshots"]
    return all(fingerprint(Path(p)) == value for p, value in snapshots.items())


def prepare():
    configs = BASE / "configs"
    configs.mkdir(exist_ok=False)
    preset_path = ROOT / "pyspi/workflow/retained185.yaml"
    preset = yaml.safe_load(preset_path.read_text())
    selected = {}
    for module, classes in preset.items():
        for name, spec in classes.items():
            if name in ("Covariance", "Precision", "SpearmanR", "CoherencePhase",
                        "CoherenceMagnitude", "PhaseLagIndex"):
                spec = dict(spec)
                if name in ("Covariance", "Precision"):
                    spec["configs"] = [p for p in spec["configs"]
                                       if p.get("estimator") == "EmpiricalCovariance"]
                selected.setdefault(module, {})[name] = spec
    count = sum(len(spec["configs"]) for c in selected.values() for spec in c.values())
    assert count == 39
    (BASE / "metrics.yaml").write_text(yaml.safe_dump(selected, sort_keys=False))
    snapshots, selections = {}, {}
    for patient in dict.fromkeys(CASES.values()):
        edf = Path(f"/media/dan/Data/data/baseline_patients/baseline_edfs/{patient}_Baseline.EDF")
        store = Path(f"/media/dan/Data/outputs/calculated_seeg_patient_data/{patient}.zarr")
        files = [edf, store / "zarr.json", *sorted((store / "patient_info/electrode").rglob("*"))]
        snapshots.update({str(p): fingerprint(p) for p in files if p.is_file()})
        selections[patient] = export_selection(store, edf, configs / f"selection-{patient}.json",
                                               patient_id=patient, recording_id="baseline")["recordings"][0]
    for channels, patient in CASES.items():
        source = selections[patient]
        assert len(source["channels"]) >= channels
        cfg = {"version": 1, "recordings": [{"path": source["path"], "patient_id": patient,
               "recording_id": f"channels-{channels}", "channels": source["channels"][:channels]}],
               "metrics": {"config": "../metrics.yaml"}, "sampling_rate_hz": 2048,
               "bandpass": {"low_hz": .5, "high_hz": 500}, "reference": "car",
               "notch": {"enabled": True, "line_frequency_hz": 60, "width_hz": 1,
                         "transition_hz": 1},
               "preprocessing": {"detrend": True, "zscore": True},
               "window": {"length_seconds": .5, "step_seconds": .5},
               "output_directory": f"../work/case-{channels}", "output_batch_mib": 64}
        (configs / f"case-{channels}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    dump(BASE / "inputs.json", {"source_snapshots": snapshots, "metric_variants": count,
         "retained_preset_sha256": sha256(preset_path.read_bytes()).hexdigest(),
         "selection": "Stored electrode order; 50 is the first 50 of patient 001's 118; 118 and 249 use the full stored selections.",
         "windows": WINDOWS, "output_location": str(BASE / "work"),
         "output_drive_matches_source_data": os.stat(BASE).st_dev == os.stat(Path(next(iter(snapshots)))).st_dev})
    assert sources_unchanged()
    print("Prepared 39 retained variants and three bounded cases.", flush=True)


def measure(channels):
    cfg_path = BASE / "configs" / f"case-{channels}.yaml"
    cfg = load_config(cfg_path)
    assert not cfg.output_directory.exists(), "Refusing to overwrite an existing trial"
    assert sources_unchanged()
    measured = Counter()
    expected = []
    finished = [0]
    original_prepare, original_commit, original_compute = (
        runner.prepare_recording, runner._commit_batch, Calculator.compute)

    def preparation(*args, **kwargs):
        print(f"{channels}: loading and preparing continuous recording", file=PROGRESS, flush=True)
        start = perf_counter()
        result = original_prepare(*args, **kwargs)
        measured["load_and_prepare_seconds"] += perf_counter() - start
        print(f"{channels}: preparation complete", file=PROGRESS, flush=True)
        return result

    def compute(calc, *args, **kwargs):
        callback = kwargs["on_metric_complete"]
        def complete(report, values):
            start = perf_counter()
            try:
                return callback(report, values)
            finally:
                measured["output_callback_seconds"] += perf_counter() - start
                finished[0] += 1
                if finished[0] % 39 == 0:
                    print(f"{channels}: window {finished[0] // 39}/{WINDOWS} calculated",
                          file=PROGRESS, flush=True)
        kwargs["on_metric_complete"] = complete
        start = perf_counter()
        try:
            return original_compute(calc, *args, **kwargs)
        finally:
            measured["calculator_compute_seconds_including_callback"] += perf_counter() - start

    def commit(path, manifest, batch, buffers, *args, **kwargs):
        start = perf_counter()
        result = original_commit(path, manifest, batch, buffers, *args, **kwargs)
        measured["batch_commit_seconds"] += perf_counter() - start
        start = perf_counter()
        expected.append({"start": batch["start"], "stop": batch["stop"],
                         "hashes": {metric: digest(value) for metric, value in buffers.items()}})
        measured["verification_capture_seconds"] += perf_counter() - start
        return result

    with (BASE / f"case-{channels}.log").open("x") as log:
        with patch.object(runner, "prepare_recording", preparation), \
             patch.object(runner, "_commit_batch", commit), \
             patch.object(Calculator, "compute", compute):
            start = perf_counter()
            with redirect_stdout(log), redirect_stderr(log):
                [path] = run(cfg_path, max_windows=WINDOWS)
            total = perf_counter() - start
    workflow_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reader = open_run(path)
    manifest = json.loads((path / "run.json").read_text())
    assert len(reader.metrics) == 39 and manifest["windows"]["count"] == WINDOWS
    reports = list(reader.diagnostics())
    assert len(reports) == 39 * WINDOWS
    assert all(e["report"]["status"] == "success" for e in reports), "Calculation failure in baseline"
    arrays, validity = {}, {}
    for metric in reader.metrics:
        values, available = reader.load(metric, return_availability=True)
        assert available.all() and values.shape == (channels, channels, WINDOWS)
        for reference in expected:
            assert digest(values[:, :, reference["start"]:reference["stop"]]) == reference["hashes"][metric]
        arrays[metric] = values
        validity[metric] = {"finite": int(np.isfinite(values).sum()), "nan": int(np.isnan(values).sum()),
                            "infinite": int(np.isinf(values).sum())}
        assert all(np.isnan(np.diag(values[:, :, w])).all() for w in range(WINDOWS))
        assert not np.isinf(values).any()
    session = manifest["sessions"][0]
    phases = {"load_and_prepare_seconds": measured["load_and_prepare_seconds"],
              "window_preprocessing_seconds": session["window_preprocessing_seconds"],
              "calculation_and_orchestration_excluding_callback_seconds": measured["calculator_compute_seconds_including_callback"] - measured["output_callback_seconds"],
              "output_callback_seconds": measured["output_callback_seconds"],
              "batch_commit_seconds": measured["batch_commit_seconds"],
              "verification_capture_seconds": measured["verification_capture_seconds"]}
    phases["setup_journal_and_other_seconds"] = total - sum(phases.values())
    assert all(v >= 0 for v in phases.values())
    assert abs(sum(phases.values()) - total) < 1e-7
    files = [p for p in cfg.output_directory.rglob("*") if p.is_file()]
    # Compare write-only batch choices using the exact same calculated values.
    # Readback checks occur outside each write timer; calculation is not repeated.
    batch_trials = []
    for batch_windows in (1, WINDOWS):
        output = cfg.output_directory / f"writer-comparison-{batch_windows}"
        started = perf_counter()
        for first in range(0, WINDOWS, batch_windows):
            write_results(output / str(first), {m: a[:, :, first:first+batch_windows] for m, a in arrays.items()})
        write_seconds = perf_counter() - started
        for metric, original in arrays.items():
            restored = np.concatenate([open_results(output / str(first)).load(metric)
                                       for first in range(0, WINDOWS, batch_windows)], axis=2)
            assert digest(restored) == digest(original)
        saved = [p for p in output.rglob("*") if p.is_file()]
        batch_trials.append({"batch_windows": batch_windows, "write_seconds": write_seconds,
                             "files": len(saved), "bytes": sum(p.stat().st_size for p in saved),
                             "all_metrics_bit_exact": True,
                             "scope": "Production writer, same real calculated arrays, no workflow journal or run metadata"})
    assert sources_unchanged()
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    receipt = {"schema": "pyspi-workflow-baseline/1", "case_channels": channels,
        "patient_id": CASES[channels], "windows": WINDOWS, "metric_variants": len(reader.metrics),
        "date_utc": datetime.now(timezone.utc).isoformat(), "observed_total_seconds": total,
        "total_excluding_verification_capture_seconds": total - measured["verification_capture_seconds"],
        "nonoverlapping_phases": phases, "original_workflow_session_fields": session,
        "per_metric_reports": reports, "all_reports_success": True,
        "all_saved_values_match_writer_input_bit_exactly": True, "output_validity": validity,
        "source_stats_unchanged": True, "workflow_peak_process_rss_kib": workflow_peak,
        "workflow_files": len(files), "workflow_bytes": sum(p.stat().st_size for p in files),
        "batch_count": len(manifest["batches"]), "writer_batch_trials": batch_trials,
        "preparation": manifest["preparation"],
        "environment": {"python": sys.version, "cpu": cpu, "logical_cpus": os.cpu_count(),
            "memory_total_kib": int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))),
            "threads": threadpool_info(), "thread_environment": {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            "versions": {p: version(p) for p in ('numpy','scipy','pandas','mne','spectral-connectivity','scikit-learn','zarr','numcodecs','pcodec','jpype1')},
            "git_commit": subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},
        "config": str(cfg_path.relative_to(ROOT)), "run": str(path.relative_to(ROOT)),
        "limits": "One serial trial per size, three initial windows, 39 selected retained variants. Full continuous EEG preparation. All estimator mathematics unchanged. Exact storage checks do not establish scientific correctness. Per-process high-water RSS includes imports/JVM/preparation/calculation/output, excludes later verification and writer comparisons. Original session fields overlap; use nonoverlapping_phases or observed total. No cohort or full-185 runtime projection."}
    dump(BASE / f"case-{channels}.json", receipt)
    print(json.dumps({"channels": channels, "total_seconds": total,
          "phases": phases, "workflow_peak_rss_gib": workflow_peak/1024**2,
          "writer_batch_trials": batch_trials, "all_reports_success": True,
          "all_saved_values_bit_exact": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--case", type=int, choices=CASES)
    args = parser.parse_args()
    prepare() if args.prepare else measure(args.case)
