"""Peak memory of the common access patterns, one fresh process per measurement.

For a symmetric and a directed metric of the largest patient, compares the current store (read
directly from the read-only source) with a packed pspi_store copy written under runs/memory:
``load`` (full n x n x W array) and ``mean`` (NaN-ignoring time average), each under default glibc
malloc settings and with a fixed mmap threshold. Peak RSS above the
post-import resident size is the kernel's high-water mark (VmHWM, reset through clear_refs), so it
includes every temporary the call creates.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")
PATIENT = "111"
METRICS = ("cov_EmpiricalCovariance", "te_symbolic_k-1_kt-1_l-1_lt-1")
STORE = HERE / "runs" / "memory" / f"{PATIENT}.pspi"
# glibc keeps freed large buffers mapped under its default dynamic mmap threshold; a fixed
# threshold returns them, which separates live data from allocator retention.
MALLOC = {"default": {}, "mmap_threshold_1MiB": {"MALLOC_MMAP_THRESHOLD_": str(1 << 20)}}


def child(fmt, op, metric, malloc):
    import numpy as np
    import zarr

    from pspi_store import open_results
    Path("/proc/self/clear_refs").write_text("5")          # reset the peak-RSS high-water mark
    base = _status_kb("VmRSS")
    t = time.perf_counter()
    if fmt == "current":
        a = zarr.open_array(str(SOURCE / f"{PATIENT}.zarr/connectivity/0.5s/{metric}"), mode="r")
        out = a[:]
        if op == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = np.nanmean(out, axis=2)
    else:
        r = open_results(STORE)
        out = r.load(metric) if op == "load" else r.mean(metric)
    s = time.perf_counter() - t
    peak = _status_kb("VmHWM")
    print(json.dumps(dict(format=fmt, op=op, metric=metric, malloc=malloc, seconds=s, result_MB=out.nbytes / 1e6,
                          peak_above_baseline_MB=(peak - base) / 1024)))


def _status_kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise KeyError(field)


def main():
    import zarr

    from pspi_store import write_metric
    shutil.rmtree(STORE, ignore_errors=True)
    for m in METRICS:
        src = zarr.open_array(str(SOURCE / f"{PATIENT}.zarr/connectivity/0.5s/{m}"), mode="r")
        write_metric(STORE, m, src[:])
    rows = []
    for m in METRICS:
        for fmt, op, malloc in itertools.product(("current", "packed"), ("load", "mean"), MALLOC):
            env = {**os.environ, **MALLOC[malloc]}
            r = subprocess.run([sys.executable, "-W", "ignore", __file__, "--child", fmt, op, m, malloc],
                               capture_output=True, text=True, check=True, cwd=HERE, env=env)
            row = json.loads(r.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = HERE / "reports" / "memory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory.json").write_text(json.dumps(rows, indent=1))
    shutil.rmtree(STORE.parent)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:6])
    else:
        main()
