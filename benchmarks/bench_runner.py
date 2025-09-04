import argparse
import json
import math
import os
import platform
import random
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
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
            # Extract the identifier from resulting table columns (only one SPI)
            if identifier is None and hasattr(calc, "table"):
                try:
                    identifier = list(calc.table.columns.levels[0])[0]
                except Exception:
                    identifier = None
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
    results = []
    totals = {"tested": 0, "ok": 0, "failed": 0, "timeout": 0, "skipped": 0}

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

                # Skip if dependencies are missing will be handled by Calculator.
                failed_or_timeout = False
                for (c, t) in shapes:
                    if failed_or_timeout:
                        break
                    attempt_note = None

                    micro_cfg = _build_micro_config(module, measure, entry, cfg_idx)
                    if dry_run:
                        status = "ok"
                        times = [0.0]
                        mean = 0.0
                        std = 0.0
                        ident = f"{measure}(config={cfg_idx})"
                    else:
                        q: Queue = get_context("spawn").Queue()
                        p = Process(
                            target=_child_worker,
                            args=(micro_cfg, c, t, repeats, max_seconds, q),
                        )
                        p.start()
                        p.join(timeout=max_seconds + 5.0)

                        if p.is_alive():
                            # Hard timeout; terminate
                            p.terminate()
                            p.join(2.0)
                            status = "timeout"
                            times = []
                            mean = None
                            std = None
                            ident = None
                            failed_or_timeout = True
                        else:
                            try:
                                child_out = q.get_nowait()
                            except Exception:
                                child_out = {"status": "failed", "error": "no output from child"}
                            status = child_out.get("status", "failed")
                            times = child_out.get("times_sec", [])
                            mean = child_out.get("mean")
                            std = child_out.get("std")
                            ident = child_out.get("identifier")
                            if status in ("failed", "timeout"):
                                failed_or_timeout = True
                                attempt_note = child_out.get("error")

                    attempt = Attempt(
                        c=c,
                        t=t,
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
                all_status = [a["status"] for a in record["attempts"]]
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
                totals["tested"] += 1
                record["identifier"] = ident
                results.append(record)

    out = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_path": os.path.abspath(config_path),
            "hardware": _hardware_info(),
            "shapes_tested": [{"c": c, "t": t, "repeats": repeats} for (c, t) in shapes],
            "max_seconds": max_seconds,
        },
        "results": results,
        "totals": totals,
    }
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


def main():
    ap = argparse.ArgumentParser(description="Benchmark pyspi measurement/config runtime")
    ap.add_argument("--config", required=True, help="Path to FullConfig.yaml or similar")
    ap.add_argument("--shapes", default=None, help="Comma list c:t pairs, e.g., 2:10,3:20,4:30")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--modules", default=None, help="Comma-separated module filters (e.g., .statistics.basic)")
    ap.add_argument("--measures", default=None, help="Comma-separated measure names to include (e.g., PairwiseDistance,DistanceCorrelation)")
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shapes = parse_shapes(args.shapes) if args.shapes else DEFAULT_SHAPES
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
