"""Full-size container benchmark: every metric and window of selected patients (read-only source).

For each metric the source array is loaded once; each container variant then writes it
(buffered, then fsync'd), evicts its own files from the page cache, and is timed on a cold
full read, a cold time-mean, a warm full read, and a warm single-window read. All exact
variants are validated bit-for-bit (values and time means). Rows go to
reports/<run>/containers.jsonl; outputs are deleted after each patient unless --keep.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import psutil
import zarr

from codecs_lib import ALL
from containers import DEFAULT_CONTAINERS, all_containers
from packing import pack

HERE = Path(__file__).resolve().parent
SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def nanmean_w(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(d.astype(np.float64, copy=False), axis=2)


def unit_files(path: Path):
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.is_file()]


def fsync_files(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    dirs = {f.parent for f in files}
    for d in dirs:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def evict(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def sizes(files):
    st = [f.stat() for f in files]
    return dict(bytes=sum(s.st_size for s in st), alloc=sum(s.st_blocks * 512 for s in st), files=len(files))


def residual_decider(codec="pcodec8"):
    enc = ALL[codec][0]

    def ok(res, full):
        a = sum(len(enc(v)) for v in res.arrays().values())
        b = sum(len(enc(v)) for v in full.arrays().values())
        return a < b
    return ok


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


def corruption_test(container, unit_root: Path, metric: str, reference: np.ndarray, scratch: Path, exact: bool):
    """Flip one byte at several offsets of the largest data file and classify what a reader sees."""
    src = container.path(unit_root, metric)
    files = unit_files(src)
    data_files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
    target_rel = data_files[0].relative_to(src) if src.is_dir() else None
    results = []
    w = reference.shape[2]
    for frac in (0.25, 0.5, 0.75, "meta"):
        work = scratch / f"corrupt_{container.name}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        dst = container.path(work, metric)
        if src.is_dir():
            shutil.copytree(src, dst)
            if frac == "meta":
                metas = [f for f in unit_files(dst) if f.name in ("zarr.json", "meta.json")]
                target = sorted(metas)[0] if metas else dst / target_rel
            else:
                target = dst / target_rel
        else:
            shutil.copy2(src, dst)
            target = dst
        size = target.stat().st_size
        off = size - 20 if frac == "meta" and not src.is_dir() else (size // 2 if frac == "meta" else int(size * frac))
        with open(target, "r+b") as fh:
            fh.seek(off)
            b = fh.read(1)
            fh.seek(off)
            fh.write(bytes([b[0] ^ 0x10]))
        outcome = dict(position=str(frac), file=str(target.relative_to(work)), offset=off, file_bytes=size)
        try:
            back = container.read_metric(work, metric)
            if exact:
                bad = [i for i in range(w) if not same_bits(np.ascontiguousarray(back[:, :, i]), np.ascontiguousarray(reference[:, :, i]))]
            else:
                bad = []
            outcome.update(full_read="silent_corruption" if bad else "ok", silent_bad_windows=len(bad))
        except Exception as e:  # noqa: BLE001 - classification of reader failures is the point
            outcome.update(full_read="error", error=f"{type(e).__name__}: {str(e)[:120]}")
        # per-window salvage
        good = err = silent = 0
        for i in range(w):
            try:
                one = container.read_metric(work, metric, windows=[i])
                if not exact or same_bits(np.ascontiguousarray(one[:, :, 0]), np.ascontiguousarray(reference[:, :, i])):
                    good += 1
                else:
                    silent += 1
            except Exception:  # noqa: BLE001
                err += 1
        outcome.update(windows_recoverable=good, windows_error=err, windows_silent=silent, windows=w)
        results.append(outcome)
        shutil.rmtree(work)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--patients", nargs="+", default=["004", "105"])
    ap.add_argument("--containers", nargs="+", default=list(DEFAULT_CONTAINERS), choices=list(all_containers()))
    ap.add_argument("--variants", nargs="+", default=["f64:exact"],
                    help="precision:structure pairs, e.g. f64:exact f64:tol f32:exact")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--metrics", nargs="+")
    ap.add_argument("--max-metrics", type=int)
    ap.add_argument("--corrupt-metric", default="cov_EmpiricalCovariance")
    ap.add_argument("--no-corrupt", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--zarr-threads", type=int, default=1)
    args = ap.parse_args()
    zarr.config.set({"threading.max_workers": args.zarr_threads, "async.concurrency": 10})
    containers = all_containers()
    report = HERE / "reports" / args.run
    report.mkdir(parents=True, exist_ok=True)
    env = dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
               logical_cpus=os.cpu_count(), memory=psutil.virtual_memory().total,
               packages={k: version(k) for k in ("numpy", "zarr", "numcodecs", "pcodec", "blosc2", "pyarrow", "zstandard")},
               args=vars(args), timing="single-threaded codecs; zarr threading.max_workers as given; "
               "cold = after fsync + POSIX_FADV_DONTNEED of the variant's own files; warm = immediately repeated")
    (report / "containers_env.json").write_text(json.dumps(env, indent=1))
    rows_path = report / "containers.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if "metric" in r:
                done.add((r["patient"], r["metric"], r["variant"]))
    scratch = HERE / "runs" / args.run / "_scratch"
    proc = psutil.Process()
    with rows_path.open("a") as fh:
        for patient in args.patients:
            group = SOURCE / f"{patient}.zarr" / "connectivity" / "0.5s"
            metrics = sorted(p.name for p in group.iterdir() if (p / "zarr.json").is_file()
                             and len(json.loads((p / "zarr.json").read_text()).get("shape", [])) == 3)
            if args.metrics:
                metrics = [m for m in metrics if m in args.metrics]
            if args.max_metrics:
                metrics = metrics[: args.max_metrics]
            variants = [(c, v) for v in args.variants for c in args.containers]
            roots = {}
            for c, v in variants:
                root = HERE / "runs" / args.run / patient / f"{c}@{v.replace(':', '-')}"
                root.mkdir(parents=True, exist_ok=True)
                roots[(c, v)] = root
            t_patient = time.perf_counter()
            for mi, metric in enumerate(metrics, 1):
                src = group / metric
                labels = json.loads((src / "zarr.json").read_text()).get("attributes", {}).get("pyspi_labels", [])
                d64, load_s = timed(lambda: np.ascontiguousarray(zarr.open_array(str(src), mode="r")[:]))
                for v in args.variants:
                    prec, structure = v.split(":")
                    d = d64 if prec == "f64" else d64.astype(np.float32)
                    expected_mean = nanmean_w(d)
                    p, pack_s = timed(pack, d, structure=structure, tolerance=args.tol, residual_ok=residual_decider())
                    exact = not p.info.get("tolerance_applied")
                    for c in args.containers:
                        key = f"{c}@{v}"
                        if (patient, metric, key) in done:
                            continue
                        cont, root = containers[c], roots[(c, v)]
                        unit = cont.path(root, metric)
                        if unit.exists():
                            shutil.rmtree(unit) if unit.is_dir() else unit.unlink()
                        extra = dict(metric=metric, pyspi_labels=labels, source_patient=patient)
                        rss0 = proc.memory_info().rss
                        _, write_s = timed(cont.write_metric, root, metric, p, extra)
                        files = unit_files(unit)
                        _, fsync_s = timed(fsync_files, files)
                        sz = sizes(files)
                        evict(files)
                        back, cold_read_s = timed(cont.read_metric, root, metric)
                        if exact and not same_bits(back, d):
                            raise AssertionError(f"{key} {patient}/{metric}: read not bit-exact")
                        if not exact:
                            iu = np.triu_indices(p.n)
                            if not same_bits(np.ascontiguousarray(back[iu[0], iu[1]]), np.ascontiguousarray(d[iu[0], iu[1]])):
                                raise AssertionError(f"{key}: kept triangle changed")
                        del back
                        evict(files)
                        mean, cold_mean_s = timed(cont.read_mean, root, metric)
                        if exact and not same_bits(mean, expected_mean):
                            diff = np.nanmax(np.abs(mean - expected_mean))
                            raise AssertionError(f"{key} {patient}/{metric}: mean not bit-exact (max diff {diff})")
                        mean_err = 0.0 if exact else float(np.nanmax(np.abs(mean - expected_mean)))
                        _, warm_read_s = timed(cont.read_metric, root, metric)
                        one, window_s = timed(cont.read_metric, root, metric, windows=[d.shape[2] // 2])
                        if exact and not same_bits(np.ascontiguousarray(one[:, :, 0]), np.ascontiguousarray(d[:, :, d.shape[2] // 2])):
                            raise AssertionError(f"{key}: window subset mismatch")
                        row = dict(patient=patient, metric=metric, variant=key, container=c, precision=prec,
                                   structure=structure, mode=p.mode, n=p.n, windows=p.windows,
                                   raw_f64_bytes=d64.nbytes, load_s=load_s, pack_s=pack_s,
                                   write_s=write_s, fsync_s=fsync_s, cold_read_s=cold_read_s,
                                   cold_mean_s=cold_mean_s, warm_read_s=warm_read_s, window_read_s=window_s,
                                   tol_applied=not exact, tol_mean_max_abs_err=mean_err,
                                   rss_delta=proc.memory_info().rss - rss0, **sz)
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        if not args.no_corrupt and metric == args.corrupt_metric and v == args.variants[0]:
                            res = corruption_test(cont, root, metric, d, scratch, exact)
                            fh.write(json.dumps(dict(patient=patient, corruption=key, results=res)) + "\n")
                            fh.flush()
                del d64
                if mi % 10 == 0 or mi == len(metrics):
                    print(f"{patient} {mi}/{len(metrics)} {metric} {time.perf_counter() - t_patient:.0f}s", flush=True)
            # per-patient metadata files outside metric units (none currently) and cleanup
            if not args.keep:
                shutil.rmtree(HERE / "runs" / args.run / patient)


if __name__ == "__main__":
    main()
