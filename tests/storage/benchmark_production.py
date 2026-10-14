"""Bounded synthetic storage timings, including packing, codec selection, and fsync.

Uses installed pyspi; never opens patient data. Timings are warm-cache local
measurements, not cohort throughput estimates. Component timings overlap:
compression selection is included in packing, which is included in write time.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pyspi import storage
from pyspi.storage import _store


@contextmanager
def component_timing(samples):
    originals = {name: getattr(_store, name) for name in
                 ("pack", "_residual_smaller", "sync_tree", "write_manifest")}
    def timed(name, function):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                samples[name] = samples.get(name, 0.0) + time.perf_counter() - start
        return call
    def selector(*args, **kwargs):
        return timed("compression_selection", originals["_residual_smaller"](*args, **kwargs))
    try:
        for name in ("pack", "sync_tree", "write_manifest"):
            setattr(_store, name, timed(name, originals[name]))
        _store._residual_smaller = selector
        yield
    finally:
        for name, function in originals.items():
            setattr(_store, name, function)


def check(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def run(repetitions):
    rng = np.random.default_rng(20261014)
    rows = []
    # Maximum full input here is 118*118*600*8 = 66.8 MB, one metric at a time.
    for n, windows in ((50, 64), (118, 96), (300, 32), (118, 600)):
        for kind in ("full", "sym", "sym_res"):
            data = rng.standard_normal((n, n, windows))
            if kind != "full":
                i, j = np.triu_indices(n, 1)
                data[j, i] = data[i, j]
            if kind == "sym_res":
                data[1, 0] = np.nextafter(data[1, 0], np.inf)
            for codec in ("pcodec", "zstd"):
                samples = []
                for _ in range(repetitions):
                    with tempfile.TemporaryDirectory(prefix="pyspi-storage-bench-") as temporary:
                        path = Path(temporary) / "results"
                        times = {}
                        start = time.perf_counter()
                        with component_timing(times):
                            meta = storage.write_metric(path, "metric", data, codec=codec)
                        times["write_total"] = time.perf_counter() - start
                        times.setdefault("compression_selection", 0.0)
                        reader = storage.open_results(path)
                        for task, selection in (("read_all", None), ("read_subset", [windows - 1, 0, 3, 0])):
                            start = time.perf_counter()
                            actual = reader.load("metric", selection)
                            times[task] = time.perf_counter() - start
                            check(actual, data if selection is None else data[..., selection])
                        start = time.perf_counter()
                        assert storage.verify(path) == {}
                        times["verify"] = time.perf_counter() - start
                        disk_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                        samples.append(times)
                row = dict(channels=n, windows=windows, metrics=1, input_kind=kind,
                           codec=codec, mode=meta["mode"], input_bytes=data.nbytes,
                           file_bytes=disk_bytes, samples_seconds=samples,
                           median_seconds={key: statistics.median(s[key] for s in samples) for key in samples[0]})
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples_seconds"}), flush=True)
    batch_samples = []
    data = rng.standard_normal((4, 4, 7))
    batch = {f"metric_{i:04d}": data for i in range(185)}
    for _ in range(repetitions):
        with tempfile.TemporaryDirectory(prefix="pyspi-storage-batch-") as temporary:
            path = Path(temporary) / "results"
            start = time.perf_counter()
            storage.write_results(path, batch)
            batch_samples.append(time.perf_counter() - start)
            assert storage.open_results(path).metrics == list(batch)
            assert storage.verify(path) == {}
            check(storage.open_results(path).load("metric_0184"), data)
            batch_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return dict(python=platform.python_version(), platform=platform.platform(),
                logical_cpus=os.cpu_count(), cpu_model=next((line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
                package_origin=str(Path(_store.__file__).resolve()),
                packages={name: importlib.metadata.version(name) for name in
                          ("pyspi", "numpy", "zarr", "numcodecs", "pcodec", "google-crc32c")},
                repetitions=repetitions, temporary_directory=tempfile.gettempdir(),
                method=__doc__, rows=rows,
                batch=dict(channels=4, windows=7, metrics=185, codec="pcodec", file_bytes=batch_bytes,
                           samples_seconds=batch_samples, median_seconds=statistics.median(batch_samples)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    result = run(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
