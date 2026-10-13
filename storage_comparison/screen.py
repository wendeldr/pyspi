"""In-memory codec x structure x layout x precision screen on existing results (read-only source).

Each task loads one patient/metric window slice, then for every combination encodes and
decodes the payload, validates bit-exact roundtrips (or recorded tolerance bounds), and
appends one JSON row per combination to ``reports/<run>/screen.jsonl``. Re-running with the
same run name skips completed tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import zarr

from codecs_lib import ALL, FAST
from packing import pack, unpack

HERE = Path(__file__).resolve().parent
SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def nanmean_w(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(d.astype(np.float64, copy=False), axis=2)


def precision_stats(d64, d32):
    a = d32.astype(np.float64)
    fin = np.isfinite(d64)
    err = np.abs(a - d64)[fin & np.isfinite(a)]
    scale = np.abs(d64)[fin & np.isfinite(a)]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, err / scale, 0.0)
    m64, m32 = nanmean_w(d64), nanmean_w(d32)
    mf = np.isfinite(m64) & np.isfinite(m32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mrel = np.where(np.abs(m64[mf]) > 0, np.abs(m32[mf] - m64[mf]) / np.abs(m64[mf]), 0.0)
    span = np.nanmax(np.abs(m64)) if np.isfinite(m64).any() else 0.0
    return dict(
        f32_max_abs_err=float(err.max()) if err.size else 0.0,
        f32_max_rel_err=float(rel.max()) if rel.size else 0.0,
        f32_mean_max_rel_err=float(mrel.max()) if mrel.size else 0.0,
        f32_mean_max_abs_err=float(np.abs(m32[mf] - m64[mf]).max()) if mf.any() else 0.0,
        f32_mean_abs_err_over_maxabs=float(np.abs(m32[mf] - m64[mf]).max() / span) if mf.any() and span else 0.0,
        f32_new_inf=int(np.sum(fin & ~np.isfinite(a))),
        f32_flushed_zero=int(np.sum(fin & (d64 != 0) & (a == 0))),
        f32_subnormal=int(np.sum(fin & (a != 0) & (np.abs(a) < np.finfo(np.float32).tiny))),
        f32_nan_changed=int(np.sum(np.isnan(d64) != np.isnan(a))),
    )


def encode_arrays(arrays, codec, layout, repeats):
    enc, dec = ALL[codec]
    size = 0
    enc_s = dec_s = 0.0
    for key, arr in arrays.items():
        a = np.ascontiguousarray(arr.T) if layout == "pm" else arr
        best_e = best_d = float("inf")
        for _ in range(repeats):
            t = time.perf_counter(); blob = enc(a); best_e = min(best_e, time.perf_counter() - t)
            t = time.perf_counter(); back = dec(blob, a.dtype, a.shape); best_d = min(best_d, time.perf_counter() - t)
        if not same_bits(np.ascontiguousarray(back), a):
            raise AssertionError(f"{codec}/{layout}/{key}: lossy roundtrip")
        size += len(blob)
        enc_s += best_e
        dec_s += best_d
    return size, enc_s, dec_s


def run_task(task):
    patient, metric, start, stop, codecs, structures, layouts, precisions, tol, repeats = task
    path = SOURCE / f"{patient}.zarr" / "connectivity" / "0.5s" / metric
    meta = json.loads((path / "zarr.json").read_text())
    labels = meta.get("attributes", {}).get("pyspi_labels", [])
    t = time.perf_counter()
    d64 = np.ascontiguousarray(zarr.open_array(str(path), mode="r")[:, :, start:stop])
    load_s = time.perf_counter() - t
    n, _, w = d64.shape
    raw = d64.nbytes
    base = dict(patient=patient, metric=metric, n=n, windows=w, raw_bytes=raw, load_s=load_s,
                directed="directed" in labels, undirected="undirected" in labels)
    rows = []
    for prec in precisions:
        d = d64 if prec == "f64" else d64.astype(np.float32)
        pstats = precision_stats(d64, d) if prec == "f32" else {}
        full_results = {}
        for structure in structures:
            t = time.perf_counter()
            p = pack(d, structure="full" if structure == "full" else structure, tolerance=tol,
                     residual_ok=lambda res, full: True)
            pack_s = time.perf_counter() - t
            t = time.perf_counter()
            back = unpack(p)
            unpack_s = time.perf_counter() - t
            if structure == "tol" and p.info.get("tolerance_applied"):
                # Values in the kept triangle are exact; mirrored values differ by <= recorded asymmetry.
                iu = np.triu_indices(n)
                if not same_bits(np.ascontiguousarray(back[iu[0], iu[1]]), np.ascontiguousarray(d[iu[0], iu[1]])):
                    raise AssertionError(f"{patient}/{metric}: tolerance pack changed kept triangle")
            elif not same_bits(back, d):
                raise AssertionError(f"{patient}/{metric}/{prec}/{structure}: pack roundtrip not bit-exact")
            del back
            for layout in layouts:
                for codec in codecs:
                    size, enc_s, dec_s = encode_arrays(p.arrays(), codec, layout, repeats)
                    mode = p.mode
                    if structure == "full":
                        full_results[(layout, codec)] = (size, enc_s, dec_s)
                    elif mode.endswith("_res") and (layout, codec) in full_results:
                        # Adaptive writer: keep the residual form only when it is smaller than full.
                        fsize, fe, fd = full_results[(layout, codec)]
                        if fsize <= size:
                            size, enc_s, dec_s, mode = fsize, enc_s + fe, fd, "full"
                    rows.append(dict(base, precision=prec, structure=structure, mode=mode, layout=layout,
                                     codec=codec, bytes=size, enc_s=enc_s, dec_s=dec_s,
                                     pack_s=pack_s, unpack_s=unpack_s,
                                     tol_applied=bool(p.info.get("tolerance_applied")),
                                     max_rel_asym=p.info.get("max_rel_asym"),
                                     max_rel_antiasym=p.info.get("max_rel_antiasym"), **pstats))
    return patient, metric, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--patients", nargs="+", default=["004", "105", "111"])
    ap.add_argument("--metrics", nargs="+")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--codecs", nargs="+", default=list(FAST))
    ap.add_argument("--structures", nargs="+", default=["full", "exact", "tol"])
    ap.add_argument("--layouts", nargs="+", default=["wm", "pm"])
    ap.add_argument("--precisions", nargs="+", default=["f64", "f32"])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if "full" not in args.structures and "exact" in args.structures:
        args.structures = ["full", *args.structures]
    out = HERE / "reports" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen_args.json").write_text(json.dumps(vars(args), indent=1))
    rows_path = out / "screen.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["patient"], r["metric"]))
    tasks = []
    for patient in args.patients:
        g = SOURCE / f"{patient}.zarr" / "connectivity" / "0.5s"
        for p in sorted(g.iterdir()):
            zj = p / "zarr.json"
            if not zj.is_file():
                continue
            meta = json.loads(zj.read_text())
            if len(meta.get("shape", [])) != 3 or (args.metrics and p.name not in args.metrics):
                continue
            if (patient, p.name) in done:
                continue
            total = meta["shape"][2]
            cnt = min(args.windows, total)
            start = (total - cnt) // 2
            tasks.append((meta["shape"][0], (patient, p.name, start, start + cnt, args.codecs, args.structures,
                                             args.layouts, args.precisions, args.tol, args.repeats)))
    tasks = [t for _, t in sorted(tasks, key=lambda x: -x[0])]
    print(f"{len(tasks)} tasks ({len(done)} already done)", flush=True)
    t0 = time.perf_counter()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool, rows_path.open("a") as fh:
        for i, (patient, metric, rows) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {patient}/{metric} {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
