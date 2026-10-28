"""Compare cache/worker/thread settings in fresh processes on actual EDF input.

Provide a workflow YAML with the intended retained channels and preprocessing,
and a new --out directory. Source recordings are read-only. Completed profile
receipts are checkpoints; no finished case is silently recomputed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import yaml
    from pyspi.workflow import load_config, open_run
    from pyspi.workflow._output import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--cases", nargs="+", default=["1:1:256", "1:1:1024", "2:1:256", "1:2:256"],
                        help="workers:threads:cache-MiB")
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    # Resolve paths through the validated Configuration before relocating YAML.
    base = {"version": 1, "output_dtype": "float64", "recordings": [{"path": str(r.path), "patient_id": r.patient_id,
             "recording_id": r.recording_id, "channels": list(r.channels) if r.channels else None}
             for r in cfg.recordings], "metrics": {"config": str(cfg.metric_config)},
            "sampling_rate_hz": float(cfg.sampling_rate_hz), "bandpass": {
                "low_hz": cfg.bandpass[0], "high_hz": cfg.bandpass[1]},
            "notch": cfg.notch, "reference": cfg.reference,
            "window": {"length_seconds": float(cfg.window_seconds), "step_seconds": float(cfg.step_seconds)},
            "preprocessing": {"detrend": cfg.detrend, "zscore": cfg.zscore},
            "output_batch_mib": cfg.output_batch_mib, "output_directory": str(args.out / "results")}
    reports = []
    for case in args.cases:
        workers, threads, cache = map(int, case.split(":"))
        name = f"w{workers}-t{threads}-c{cache}"
        document = {**base, "run_name": name, "execution": {
            "workers": workers, "threads_per_worker": threads, "spectral_cache_mib": cache}}
        path = args.out / f"{name}.yaml"
        text = yaml.safe_dump(document, sort_keys=False)
        receipt = args.out / f"{name}.json"
        if receipt.exists():
            result = json.loads(receipt.read_text())
            assert result["config_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                       for p, digest in result["source_sha256"].items()), "Code changed; use a new benchmark directory"
        else:
            path.write_text(text)
            environment = dict(os.environ)
            environment.update({key: str(threads) for key in
                                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
            print(f"Starting {name}", flush=True)
            with (args.out / f"{name}.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "pyspi.workflow", "profile", str(path),
                    "--max-windows", str(args.windows), "--report", str(receipt)],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(receipt.read_text())
        reports.append(result)
        print(name, result["elapsed_seconds"], result["process_tree_memory"], flush=True)
    comparisons = []
    for result in reports[1:]:
        for baseline, candidate in zip(reports[0]["runs"], result["runs"], strict=True):
            a, b = open_run(baseline["path"]), open_run(candidate["path"])
            assert a.metrics == b.metrics
            for metric in a.metrics:
                x, y = a.load(metric), b.load(metric)
                finite = np.isfinite(x) & np.isfinite(y)
                comparisons.append({"execution": result["execution"], "metric": metric,
                    "nan_mask_equal": bool(np.array_equal(np.isnan(x), np.isnan(y))),
                    "infinity_mask_equal": bool(np.array_equal(np.isinf(x), np.isinf(y))),
                    "statuses_equal": list(a.statuses(metric)) == list(b.statuses(metric)),
                    "max_absolute_difference": float(np.max(abs(x[finite]-y[finite]))) if finite.any() else None,
                    "allclose": bool(np.allclose(x, y, rtol=1e-10, atol=1e-12, equal_nan=True))})
    atomic_json(args.out / "comparison.json", {"profiles": [str(args.out / f"w{r['execution']['workers']}-t{r['execution']['threads_per_worker']}-c{int(r['execution']['spectral_cache_mib'])}.json") for r in reports],
                                              "comparisons": comparisons})


if __name__ == "__main__":
    main()
