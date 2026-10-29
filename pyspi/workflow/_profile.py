"""Bounded real workflow measurement including preparation and durable output."""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
from time import perf_counter

from ._config import load_config
from ._output import atomic_json, read_json
from ._parallel import peak_rss_mib, jvm_started
from ._resources import ProcessTreeMemory, available_memory_mib
from ._run import run


def profile(config, *, max_windows, report_path):
    cfg = load_config(config)
    started = perf_counter()
    initial_available = available_memory_mib()
    initial_load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    with ProcessTreeMemory() as memory:
        paths = run(cfg, max_windows=max_windows)
    elapsed = perf_counter()-started
    records = []
    total_windows = 0
    for path in paths:
        manifest = read_json(path / "run.json")
        session = manifest["sessions"][-1]
        total_windows += manifest["windows"]["count"]
        records.append({"path": str(path), "channels": len(manifest["channels"]),
            "windows": manifest["windows"], "metrics": len(manifest["contract"]["metrics"]),
            "session": session})
    per_window = elapsed / total_windows if total_windows else None
    root = Path(__file__).resolve().parents[2]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((root / "pyspi").rglob("*.py"))}
    result = {"schema": "pyspi-workflow-profile/1", "configuration": str(cfg.source),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(cfg.source.read_bytes()).hexdigest(),
        "metric_config_sha256": hashlib.sha256(cfg.metric_config.read_bytes()).hexdigest(),
        "execution": cfg.execution, "python": platform.python_version(),
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "system_available_before_mib": initial_available,
        "load_average_before": initial_load,
        "load_average_after": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "elapsed_seconds": elapsed, "windows": total_windows,
        "end_to_end_seconds_per_window": per_window,
        "end_to_end_seconds_600": per_window*600 if per_window is not None else None,
        "end_to_end_seconds_41400": per_window*41400 if per_window is not None else None,
        "parent_peak_rss_mib": peak_rss_mib(), "parent_jvm_started": jvm_started(),
        "process_tree_memory": memory.as_dict(),
        "runs": records, "source_sha256": source_hashes,
        "limits": "Bounded trial, including full-record preparation, startup and output writes. End-to-end projections charge that trial's setup per window; they are not measured cohort throughput. Compare steady-state calculation stages separately. Patient conditioning and host contention can change results."}
    atomic_json(Path(report_path), result)
    return result


def profile_experiment(config, *, max_windows, report_path, resume=False):
    """Whole-process measurement of all groups; one output per original window."""
    from .experiment import load_experiment, run_experiment
    experiment = load_experiment(config)
    started = perf_counter()
    initial_available = available_memory_mib()
    initial_load = list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
    with ProcessTreeMemory() as memory:
        result = run_experiment(experiment, max_windows=max_windows, resume=resume)
    elapsed = perf_counter()-started
    records, counts = [], {}
    for entry in result['runs']:
        manifest = read_json(Path(entry['path'])/'run.json')
        key = (entry['patient_id'], entry['recording_id'])
        counts[key] = manifest['windows']['count']
        records.append({**entry, 'channels': len(manifest['channels']), 'windows': manifest['windows'],
                        'metrics': len(manifest['contract']['metrics']), 'session': manifest['sessions'][-1]})
    windows = sum(counts.values())
    per_window = elapsed/windows if windows else None
    root = Path(__file__).resolve().parents[2]
    report = {'schema': 'pyspi-experiment-profile/1', 'configuration': str(experiment.source),
              'created_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': elapsed,
              'recording_windows': windows, 'group_windows': sum(r['windows']['count'] for r in records),
              'end_to_end_seconds_per_recording_window': per_window,
              'end_to_end_seconds_600': per_window*600 if windows else None,
              'end_to_end_seconds_41400': per_window*41400 if windows else None,
              'runs': records, 'preparation_cache': result['preparation_cache'],
              'process_tree_memory': memory.as_dict(), 'parent_jvm_started': jvm_started(),
              'system_available_before_mib': initial_available, 'load_average_before': initial_load,
              'load_average_after': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((root/'pyspi').rglob('*.py'))},
              'config_sha256': hashlib.sha256(experiment.source.read_bytes()).hexdigest(),
              'metric_config_sha256': {name: hashlib.sha256(cfg.metric_config.read_bytes()).hexdigest()
                                      for name, cfg in experiment.groups},
              'limits': 'Full preparation, setup, computation and durable writing included. Per-recording-window costs include all groups, not divided by their count. Bounded trials overcharge startup in cohort projections; inspect stages and whole-process memory. Resume measurements include only newly executed work and must not be treated as full-run timing.'}
    atomic_json(Path(report_path), report)
    return report
