"""Thread scaling of write/read for sharded Zarr packed stores on a large (111-sized) metric.

Each (codec, threads) measurement runs in a fresh subprocess so zarr's thread pool is sized
from the configuration. Also measures raw codec GIL release with a thread pool over chunks.
Source is read-only; outputs go under runs/thread_scaling and are deleted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")


def child(patient, metric, codec, threads, prec):
    import numpy as np
    import zarr
    zarr.config.set({"threading.max_workers": threads, "async.concurrency": max(10, threads * 2)})
    from concurrent.futures import ThreadPoolExecutor

    from bench_containers import evict, fsync_files, unit_files
    from codecs_lib import ALL
    from containers import ZarrPacked, chunk_windows
    from packing import pack

    d = zarr.open_array(str(SOURCE / f"{patient}.zarr/connectivity/0.5s/{metric}"), mode="r")[:]
    if prec == "f32":
        d = d.astype(np.float32)
    p = pack(d, structure="exact")
    cont = ZarrPacked(codec, sharded=True)
    root = HERE / "runs" / "thread_scaling" / f"{codec}_{prec}_{threads}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    t = time.perf_counter(); cont.write_metric(root, metric, p, {}); write_s = time.perf_counter() - t
    files = unit_files(cont.path(root, metric))
    t = time.perf_counter(); fsync_files(files); fsync_s = time.perf_counter() - t
    evict(files)
    t = time.perf_counter(); back = cont.read_metric(root, metric); cold = time.perf_counter() - t
    assert back.tobytes() == d.tobytes()
    warm = min(_time(lambda: cont.read_metric(root, metric)) for _ in range(3))
    evict(files)
    t = time.perf_counter(); cont.read_mean(root, metric); cold_mean = time.perf_counter() - t
    # raw codec with a thread pool over chunks (GIL release check)
    arr = p.upper if p.upper is not None else p.full
    k = chunk_windows(arr.shape[1], arr.shape[0])
    chunks = [np.ascontiguousarray(arr[s:s + k]) for s in range(0, arr.shape[0], k)]
    enc, dec = ALL[codec]
    with ThreadPoolExecutor(threads) as ex:
        t = time.perf_counter(); blobs = list(ex.map(enc, chunks)); raw_enc = time.perf_counter() - t
        t = time.perf_counter(); list(ex.map(lambda ab: dec(ab[1], ab[0].dtype, ab[0].shape), zip(chunks, blobs))); raw_dec = time.perf_counter() - t
    shutil.rmtree(root)
    print(json.dumps(dict(patient=patient, metric=metric, codec=codec, precision=prec, threads=threads, mode=p.mode,
                          raw_f64_bytes=int(np.prod(d.shape)) * 8, write_s=write_s, fsync_s=fsync_s, cold_read_s=cold,
                          warm_read_s=warm, cold_mean_s=cold_mean, raw_pool_enc_s=raw_enc, raw_pool_dec_s=raw_dec,
                          chunks=len(chunks))))


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    patient = "111"
    out = HERE / "reports" / "thread_scaling"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in ("cov_EmpiricalCovariance",):
        for codec in ("pcodec8", "zstd3"):
            for prec in ("f64",):
                for threads in (1, 4, 16):
                    r = subprocess.run([sys.executable, "-W", "ignore", __file__, "--child", patient, metric, codec,
                                        str(threads), prec], capture_output=True, text=True, check=True, cwd=HERE,
                                       env={**os.environ, "OMP_NUM_THREADS": "1"})
                    row = json.loads(r.stdout.strip().splitlines()[-1])
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    (out / "thread_scaling.json").write_text(json.dumps(rows, indent=1))
    shutil.rmtree(HERE / "runs" / "thread_scaling", ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    else:
        main()
