"""Bounded synthetic before/after storage timings, including durable file flushes.

Run with the isolated test Python. Supply the old pspi_store.py extracted from
Git as --baseline; outputs and temporary stores stay outside patient data.
"""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr

PROTOTYPE = Path(__file__).resolve().parents[2] / "storage_comparison"
sys.path.insert(0, str(PROTOTYPE))

import pspi_store as current
from store_integrity import sync_tree


def run(baseline_path, repetitions):
    spec = importlib.util.spec_from_file_location("baseline_pspi_store", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rows = []
    rng = np.random.default_rng(20261013)
    with tempfile.TemporaryDirectory(prefix="pspi-integrity-benchmark-") as temporary:
        root = Path(temporary)
        for n, windows in ((40, 64), (118, 96), (249, 32), (118, 600)):
            directed = rng.standard_normal((n, n, windows))
            symmetric = directed.copy()
            i, j = np.triu_indices(n, 1)
            symmetric[j, i] = symmetric[i, j]
            near_symmetric = symmetric.copy()
            near_symmetric[1, 0] = np.nextafter(near_symmetric[1, 0], np.inf)
            for kind, data in (("full", directed), ("sym", symmetric), ("sym_res", near_symmetric)):
                for codec in ("pcodec", "zstd"):
                    samples = {name: {task: [] for task in ("write", "load_all", "load_subset")}
                               for name in ("before", "after")}
                    sizes, modes = {}, {}
                    for repeat in range(repetitions):
                        order = (("before", baseline), ("after", current))
                        if repeat % 2:
                            order = order[::-1]
                        for name, module in order:
                            path = root / f"{name}-{repeat}.pspi"
                            start = time.perf_counter()
                            meta = module.write_metric(path, "metric", data, codec=codec)
                            if name == "before":
                                # Original writer did not flush. Include the same
                                # file/directory flush for a durable-write comparison.
                                sync_tree(path)
                            samples[name]["write"].append(time.perf_counter() - start)
                            sizes[name] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                            modes[name] = meta["mode"]
                            reader = module.open_results(path)
                            for task, selection in (("load_all", None), ("load_subset", [0, windows - 1, 3, 0])):
                                start = time.perf_counter()
                                values = reader.load("metric", selection)
                                samples[name][task].append(time.perf_counter() - start)
                                expected = data if selection is None else data[:, :, selection]
                                assert values.dtype == expected.dtype and values.shape == expected.shape
                                assert np.ascontiguousarray(values).tobytes() == np.ascontiguousarray(expected).tobytes()
                            assert module.verify(path) == {}
                            shutil.rmtree(path)
                    row = dict(n=n, windows=windows, metrics=1, kind=kind, codec=codec, modes=modes,
                               disk_bytes=sizes, samples_seconds=samples,
                               median_seconds={name: {task: statistics.median(times) for task, times in tasks.items()}
                                               for name, tasks in samples.items()})
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "samples_seconds"}), flush=True)
        # Expose manifest growth at the requested 185-metric inventory size,
        # using tiny matrices so this remains a storage-only bounded probe.
        data = rng.standard_normal((4, 4, 7))
        i, j = np.tril_indices(4, -1)
        data[i, j] = data[j, i]
        metrics = {f"metric_{number:04d}": data for number in range(185)}
        samples = {name: [] for name in ("before", "after")}
        sizes = {}
        for repeat in range(repetitions):
            order = (("before", baseline), ("after", current))
            if repeat % 2:
                order = order[::-1]
            for name, module in order:
                path = root / f"batch-{name}-{repeat}.pspi"
                start = time.perf_counter()
                module.write_results(path, metrics)
                if name == "before":
                    sync_tree(path)
                samples[name].append(time.perf_counter() - start)
                sizes[name] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                reader = module.open_results(path)
                assert len(reader.metrics) == 185
                assert reader.load("metric_0184").tobytes() == data.tobytes()
                assert module.verify(path) == {}
                shutil.rmtree(path)
        batch = dict(n=4, windows=7, metrics=185, codec="pcodec", disk_bytes=sizes,
                     samples_seconds=samples, median_seconds={name: statistics.median(times) for name, times in samples.items()})
        print(json.dumps({"batch": batch}), flush=True)
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.processor())
    return dict(python=platform.python_version(), numpy=np.__version__, zarr=zarr.__version__,
                platform=platform.platform(), repetitions=repetitions, baseline=str(baseline_path),
                cpu_model=cpu_model, logical_cpus=os.cpu_count(), temporary_directory=tempfile.gettempdir(),
                method="alternating order; warm OS cache; original writer plus fsync; no EEG computation", rows=rows, batch=batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    result = run(args.baseline, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
