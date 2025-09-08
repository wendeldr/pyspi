import argparse
import hashlib
import json
import math
import os
import platform
import random
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from multiprocessing import Process, Queue, get_context
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import yaml

from .yaml_stream import iter_module_blocks


DEFAULT_SHAPES = [(2, 10), (3, 20), (4, 30), (6, 64), (8, 64)]


def _env_limit_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _child_worker(cfg: Dict[str, Any], c: int, t: int, repeats: int, max_seconds: float, q: Queue):
    """Run a single measurement/config across repeats with a wall-time budget.

    Returns on queue a dict with status, times, mean, std, identifier (if available).
    """
    _env_limit_threads()
    start_all = time.perf_counter()
    from pyspi.calculator import Calculator  # Lazy import inside child
    import sys
    import contextlib

    # Create a temp YAML with a single measurement/config
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False)
        tmp_path = tf.name

    rng = np.random.RandomState(12345)
    times = []
    
    # Extract identifier from config structure before running calculations
    identifier = None
    try:
        # cfg structure: {module: {measure: {configs: [...], ...}}}
        module = list(cfg.keys())[0]
        measure_dict = cfg[module]
        measure = list(measure_dict.keys())[0]
        entry = measure_dict[measure]
        identifier = _extract_identifier_from_config(module, measure, entry, 0)
    except Exception:
        identifier = None
    try:
        for r in range(repeats):
            if max_seconds is not None and (time.perf_counter() - start_all) >= max_seconds:
                break
            data = rng.randn(c, t)
            # Build calculator for this one-SPI config
            calc = Calculator(dataset=data, configfile=tmp_path)
            if getattr(calc, 'n_spis', 0) == 0:
                q.put({
                    "status": "skipped_missing_dependency",
                    "times_sec": [],
                    "identifier": None,
                })
                return
            t0 = time.perf_counter()
            # Suppress verbose stdout/stderr from underlying libraries for timing fairness
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                calc.compute()
            dt = time.perf_counter() - t0
            # Treat all-NaN output as failure and stop further repeats
            try:
                if identifier is not None:
                    col = calc.table[identifier]
                    if col.isna().all().all():
                        q.put({
                            "status": "failed",
                            "error": "all_nan_output",
                            "times_sec": times,
                            "identifier": identifier,
                        })
                        return
            except Exception:
                pass
            times.append(dt)
    except Exception as e:
        q.put({
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "times_sec": times,
            "identifier": identifier,
        })
        return
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    elapsed_all = time.perf_counter() - start_all
    if max_seconds is not None and elapsed_all >= max_seconds and len(times) == 0:
        q.put({"status": "timeout", "times_sec": [], "identifier": identifier})
        return

    status = "ok" if len(times) > 0 else "timeout"
    q.put({
        "status": status,
        "times_sec": times,
        "mean": float(np.mean(times)) if times else None,
        "std": float(np.std(times)) if times else None,
        "identifier": identifier,
    })


@dataclass
class Attempt:
    c: int
    t: int
    repeats: int
    wall_time_limit_sec: float
    status: str
    times_sec: List[float]
    mean: Optional[float]
    std: Optional[float]
    note: Optional[str] = None


def _build_micro_config(module: str, measure: str, entry: Dict[str, Any], config_idx: int) -> Dict[str, Any]:
    # Extract labels/dependencies if present
    labels = entry.get("labels")
    deps = entry.get("dependencies")
    cfg_list = entry.get("configs")
    if cfg_list is None:
        config = None
    else:
        config = cfg_list[config_idx]
    # Wrap into a one-measure config file structure
    body: Dict[str, Any] = {measure: {"configs": [config]}}
    if labels is not None:
        body[measure]["labels"] = labels
    if deps is not None:
        body[measure]["dependencies"] = deps
    return {module: body}


def _compute_config_hash(config_path: str) -> str:
    """Compute SHA-256 hash of the config file for comparison across runs."""
    sha256_hash = hashlib.sha256()
    with open(config_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _extract_identifier_from_config(module: str, measure: str, entry: Dict[str, Any], config_idx: int) -> str:
    """Extract the SPI identifier without instantiating Calculator or running compute.
    
    This duplicates the logic from Calculator._load_yaml to generate identifiers
    without requiring dependencies or data.
    """
    import importlib
    
    configs = entry.get("configs")
    if configs is None:
        params = {}
    else:
        params = configs[config_idx] if configs[config_idx] is not None else {}
    
    try:
        # Import the module to get the class
        mod = importlib.import_module(module, package='pyspi')
        spi_class = getattr(mod, measure)
        
        # Create a temporary instance just to get the identifier
        # This is what Calculator._load_yaml does on line 243
        spi = spi_class(**params)
        return spi.identifier
    except Exception as e:
        # If we can't instantiate (missing deps, etc.), create a fallback identifier
        # This mirrors what the benchmark runner would show in dry-run mode
        return f"{measure}(config={config_idx})"


def _hardware_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def run_benchmarks(
    config_path: str,
    shapes: List[Tuple[int, int]],
    repeats: int,
    max_seconds: float,
    out_json: str,
    modules_filter: Optional[List[str]] = None,
    measures_filter: Optional[List[str]] = None,
    dry_run: bool = False,
):
    _env_limit_threads()
    
    # Sort shapes by c values for smart timeout/failure prediction
    shapes = sorted(shapes, key=lambda x: x[0])
    
    # Calculate total number of configurations and estimate max runtime
    total_configs = 0
    for module, module_dict in iter_module_blocks(config_path):
        if modules_filter and module not in modules_filter:
            continue
        for measure, entry in module_dict.items():
            if measures_filter and measure not in measures_filter:
                continue
            configs = entry.get("configs")
            if configs is None:
                configs = [None]
            total_configs += len(configs)
    
    # Max runtime = total_configs * len(shapes) * (max_seconds + 5 for process overhead)
    max_runtime_seconds = total_configs * len(shapes) * (max_seconds + 5)
    
    start_time = datetime.now()
    estimated_finish_time = start_time + timedelta(seconds=max_runtime_seconds)
    
    print(f"Starting benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Total configurations to test: {total_configs}")
    print(f"Data shapes to test: {len(shapes)}")
    print(f"Maximum estimated runtime: {max_runtime_seconds/60:.1f} minutes ({max_runtime_seconds/3600:.1f} hours)")
    print(f"Estimated finish time (worst case): {estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("-" * 60)
    
    results = []
    totals = {"tested": 0, "ok": 0, "failed": 0, "timeout": 0, "skipped": 0, "predicted_skipped": 0}
    completed_configs = 0
    
    # Track which (module, measure, config_index) combinations have failed/timed out at specific c values
    # This allows us to skip higher c values when lower ones fail/timeout
    failure_patterns = {}  # key: (module, measure, config_index), value: {'failed_at_c': int, 'timeout_at_c': int}

    for module, module_dict in iter_module_blocks(config_path):
        if modules_filter and module not in modules_filter:
            continue
        # module_dict: {MeasureName: {labels, dependencies, configs: [...]}, ...}
        for measure, entry in module_dict.items():
            if measures_filter and measure not in measures_filter:
                continue
            configs = entry.get("configs")
            if configs is None:
                configs = [None]
            for cfg_idx in range(len(configs)):
                key = {"module": module, "measure": measure, "config_index": cfg_idx}
                record: Dict[str, Any] = {"key": key, "status": None, "attempts": []}
                config_key = (module, measure, cfg_idx)
                
                # Initialize failure tracking for this config
                if config_key not in failure_patterns:
                    failure_patterns[config_key] = {'failed_at_c': None, 'timeout_at_c': None}

                for (c, t) in shapes:
                    attempt_note = None
                    should_skip = False
                    skip_reason = None
                    
                    # Check if we should skip this c value based on previous failures/timeouts
                    pattern = failure_patterns[config_key]
                    if pattern['failed_at_c'] is not None and c >= pattern['failed_at_c']:
                        should_skip = True
                        skip_reason = f"skipped_failure_prediction_from_c{pattern['failed_at_c']}"
                    elif pattern['timeout_at_c'] is not None and c >= pattern['timeout_at_c']:
                        should_skip = True
                        skip_reason = f"skipped_timeout_prediction_from_c{pattern['timeout_at_c']}"

                    micro_cfg = _build_micro_config(module, measure, entry, cfg_idx)
                    # Extract identifier consistently whether dry-run or not
                    ident = _extract_identifier_from_config(module, measure, entry, cfg_idx)
                    
                    if should_skip:
                        # Create a skipped attempt based on prediction
                        if "failure" in skip_reason:
                            status = "failed"
                            attempt_note = skip_reason
                        else:  # timeout prediction
                            status = "timeout"
                            attempt_note = skip_reason
                        times = []
                        mean = None
                        std = None
                    elif dry_run:
                        status = "ok"
                        times = [0.0]
                        mean = 0.0
                        std = 0.0
                    else:
                        q: Queue = get_context("spawn").Queue()
                        p = Process(
                            target=_child_worker,
                            args=(micro_cfg, c, t, repeats, max_seconds, q),
                        )
                        # Make child daemonic so the parent won't try to
                        # implicitly join it during interpreter shutdown.
                        p.daemon = True
                        try:
                            p.start()
                            p.join(timeout=max_seconds + 5.0)

                            if p.is_alive():
                                # Hard timeout; escalate to kill if needed.
                                p.terminate()
                                p.join(2.0)
                                if p.is_alive():
                                    p.kill()
                                    p.join(2.0)
                                status = "timeout"
                                times = []
                                mean = None
                                std = None
                                ident = None
                                # Record this timeout for future predictions
                                if failure_patterns[config_key]['timeout_at_c'] is None:
                                    failure_patterns[config_key]['timeout_at_c'] = c
                            else:
                                try:
                                    child_out = q.get_nowait()
                                except Exception:
                                    child_out = {"status": "failed", "error": "no output from child"}
                                status = child_out.get("status", "failed")
                                times = child_out.get("times_sec", [])
                                mean = child_out.get("mean")
                                std = child_out.get("std")
                                child_ident = child_out.get("identifier")
                                # Use child identifier if available, otherwise use our extracted one
                                if child_ident is not None:
                                    ident = child_ident
                                if status == "failed":
                                    attempt_note = child_out.get("error")
                                    # Record this failure for future predictions
                                    if failure_patterns[config_key]['failed_at_c'] is None:
                                        failure_patterns[config_key]['failed_at_c'] = c
                                elif status == "timeout":
                                    attempt_note = child_out.get("error")
                                    # Record this timeout for future predictions
                                    if failure_patterns[config_key]['timeout_at_c'] is None:
                                        failure_patterns[config_key]['timeout_at_c'] = c
                        finally:
                            # Ensure the Queue's feeder thread is shut down to
                            # prevent hangs at interpreter exit.
                            try:
                                while True:
                                    try:
                                        _ = q.get_nowait()
                                    except Exception:
                                        break
                            except Exception:
                                pass
                            try:
                                q.close()
                            except Exception:
                                pass
                            try:
                                q.join_thread()
                            except Exception:
                                pass

                    attempt = Attempt(
                        c=int(c),
                        t=int(t),
                        repeats=repeats,
                        wall_time_limit_sec=max_seconds,
                        status=status,
                        times_sec=times,
                        mean=mean,
                        std=std,
                        note=attempt_note,
                    )
                    record["attempts"].append(asdict(attempt))

                # Summarize status for this key
                all_attempts = record["attempts"]
                all_status = [a["status"] for a in all_attempts]
                predicted_skips = [a for a in all_attempts if a.get("note") and "skipped_" in a.get("note", "")]
                
                if any(s == "failed" for s in all_status):
                    record["status"] = "failed"
                    totals["failed"] += 1
                elif any(s == "timeout" for s in all_status):
                    record["status"] = "timeout" 
                    totals["timeout"] += 1
                elif all(s == "skipped_missing_dependency" for s in all_status):
                    record["status"] = "skipped_missing_dependency"
                    totals["skipped"] += 1
                else:
                    record["status"] = "ok"
                    totals["ok"] += 1
                    
                # Count predicted skips separately for reporting
                totals["predicted_skipped"] += len(predicted_skips)
                totals["tested"] += 1
                record["identifier"] = ident
                results.append(record)
                
                completed_configs += 1
                elapsed_time = datetime.now() - start_time
                progress_pct = (completed_configs / total_configs) * 100
                avg_time_per_config = elapsed_time.total_seconds() / completed_configs
                remaining_configs = total_configs - completed_configs
                estimated_remaining_time = remaining_configs * avg_time_per_config
                estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
                
                # Show prediction info if any shapes were skipped
                predicted_count = len(predicted_skips) 
                prediction_info = f" (predicted {predicted_count} skips)" if predicted_count > 0 else ""
                print(f"Progress: {completed_configs}/{total_configs} ({progress_pct:.1f}%) | "
                      f"Elapsed: {elapsed_time.total_seconds()/60:.1f}min | "
                      f"ETA: {estimated_completion.strftime('%H:%M:%S')} | "
                      f"Last: {module}.{measure}[{cfg_idx}] -> {record['status']}{prediction_info}")

    out = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_path": os.path.abspath(config_path),
            "config_hash": _compute_config_hash(config_path),
            "hardware": _hardware_info(),
            "shapes_tested": [{"c": int(c), "t": int(t), "repeats": repeats} for (c, t) in shapes],
            "max_seconds": max_seconds,
        },
        "results": results,
        "totals": totals,
    }
    # Add summary of smart prediction benefits
    final_time = datetime.now()
    total_runtime = final_time - start_time
    print("-" * 60)
    print(f"Benchmark completed at: {final_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Total runtime: {total_runtime.total_seconds()/60:.1f} minutes")
    if totals["predicted_skipped"] > 0:
        print(f"Smart prediction saved {totals['predicted_skipped']} test runs by predicting failures/timeouts")
    print(f"Final totals: {totals}")
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def parse_shapes(s: str) -> List[Tuple[int, int]]:
    # format: "c1:t1,c2:t2,..."
    out: List[Tuple[int, int]] = []
    for part in s.split(","):
        if not part:
            continue
        c_str, t_str = part.split(":")
        out.append((int(c_str), int(t_str)))
    return out

def curved_ints(min_val=2, max_val=250):
    num_tens = int(max_val // 10)
    nums = []
    for i in range(num_tens):
        # z transitions from 10 (when i=0) to 1 (when i=num_tens-1)
        z = 10 - (9 * (i / max(1, num_tens - 1))**0.1)
        z = max(1, z)
        x = np.linspace(0, 9, int(round(z))).astype(int) + (i * 10)
        nums.extend(x)
    nums = np.array(nums)
    nums = np.clip(nums, min_val, max_val)
    nums = np.sort(np.unique(nums))
    return nums



def main():
    ap = argparse.ArgumentParser(description="Benchmark pyspi measurement/config runtime")
    ap.add_argument("--config", required=True, help="Path to FullConfig.yaml or similar")
    # ap.add_argument("--shapes", default=None, help="Comma list c:t pairs, e.g., 2:10,3:20,4:30")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--modules", default=None, help="Comma-separated module filters (e.g., .statistics.basic)")
    ap.add_argument("--measures", default=None, help="Comma-separated measure names to include (e.g., PairwiseDistance,DistanceCorrelation)")
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # first_col = curved_ints(2, 250)[::2]
    first_col = np.array([2, 10, 50, 80, 100, 150, 250])
    shapes = np.column_stack([first_col, np.full(first_col.size, 1024)])

    modules_filter = args.modules.split(",") if args.modules else None
    measures_filter = args.measures.split(",") if args.measures else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("benchmarks", "out", f"raw_timings_{ts}.json")
    run_benchmarks(
        config_path=args.config,
        shapes=shapes,
        repeats=args.repeats,
        max_seconds=args.max_seconds,
        out_json=out_path,
        modules_filter=modules_filter,
        measures_filter=measures_filter,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
