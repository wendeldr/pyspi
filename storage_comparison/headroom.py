"""How much lossless headroom is left after pcodec? A per-bit and split-value check.

Same sample as the screen (middle windows of every metric of each patient, exact structure
packing with the adaptive residual choice). For each stored float payload:

- per-bit entropy of the mantissa: the fraction of ones at each mantissa bit position, and the
  sum of the binary entropies (an order-0 bit model of what the mantissa can cost)
- split value: the high and low halves of each value's bits (32/32 for float64, 16/16 for
  float32) are compressed separately with the strongest general codecs here (xz -9e and
  zstd -19 on byte-shuffled data, pcodec level 12 on the unsigned integers). If the low half
  does not shrink, those bits are noise to every model tried, and a method beating pcodec
  could only win on the high half.

Writes reports/<run>/headroom.jsonl and prints per-patient totals.
"""

from __future__ import annotations

import argparse
import json
import lzma
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import zarr
import zstandard

from codecs_lib import ALL
from packing import pack
from pspi_store import _residual_smaller

HERE = Path(__file__).resolve().parent
SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")
MANT = {np.dtype(np.float64): 52, np.dtype(np.float32): 23}
HALF = {np.dtype(np.float64): (np.uint64, np.uint32, 32), np.dtype(np.float32): (np.uint32, np.uint16, 16)}


def shuffled(u):
    s = u.dtype.itemsize
    return np.ascontiguousarray(u.reshape(-1).view(np.uint8).reshape(-1, s).T).tobytes()


def best_size(u):
    """Smallest of xz -9e (shuffled), zstd -19 (shuffled), pcodec 12 for an unsigned integer array."""
    sizes = dict(xz9e=len(lzma.compress(shuffled(u), preset=9 | lzma.PRESET_EXTREME)),
                 zstd19=len(zstandard.ZstdCompressor(level=19).compress(shuffled(u))),
                 pcodec12=len(ALL["pcodec12"][0](u)))
    k = min(sizes, key=sizes.get)
    return sizes[k], k


def bit_stats(a):
    m = MANT[a.dtype]
    ut = HALF[a.dtype][0]
    v = a[np.isfinite(a) & (a != 0)]
    if not v.size:
        return None, None
    bits = v.view(ut)
    p = np.array([np.mean((bits >> ut(i)) & ut(1)) for i in range(m)])
    h = -(p * np.log2(np.clip(p, 1e-300, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-300, 1)))
    return float(h.sum()), int(np.sum(np.abs(p - 0.5) < 0.01))


def task(args):
    patient, metric, start, stop = args
    d64 = np.ascontiguousarray(zarr.open_array(str(SOURCE / f"{patient}.zarr/connectivity/0.5s/{metric}"),
                                               mode="r")[:, :, start:stop])
    rows = []
    for prec in ("f64", "f32"):
        d = d64 if prec == "f64" else d64.astype(np.float32)
        p = pack(d, structure="exact", residual_ok=_residual_smaller("pcodec"))
        a = np.ascontiguousarray(p.upper if p.upper is not None else p.full)
        full_t, half_t, shift = HALF[a.dtype]
        u = a.view(full_t)
        hi = (u >> full_t(shift)).astype(half_t)
        lo = (u & full_t((1 << shift) - 1)).astype(half_t)
        lo_b, lo_k = best_size(lo)
        hi_b, hi_k = best_size(hi)
        h_bits, uniform_bits = bit_stats(a)
        rows.append(dict(patient=patient, metric=metric, precision=prec, mode=p.mode, n=p.n, windows=p.windows,
                         payload_bytes=a.nbytes, pcodec8_bytes=len(ALL["pcodec8"][0](a)),
                         half_raw_bytes=lo.nbytes, lo_best_bytes=lo_b, lo_best_codec=lo_k,
                         hi_best_bytes=hi_b, hi_best_codec=hi_k, mantissa_bits=MANT[a.dtype],
                         mantissa_bit_entropy=h_bits, mantissa_uniform_bits=uniform_bits))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="headroom_w64")
    ap.add_argument("--patients", nargs="+", default=["004", "105", "111"])
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = HERE / "reports" / args.run
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for patient in args.patients:
        for p in sorted((SOURCE / f"{patient}.zarr" / "connectivity" / "0.5s").iterdir()):
            zj = p / "zarr.json"
            if not zj.is_file():
                continue
            meta = json.loads(zj.read_text())
            if len(meta.get("shape", [])) != 3:
                continue
            total = meta["shape"][2]
            cnt = min(args.windows, total)
            tasks.append((patient, p.name, (total - cnt) // 2, (total - cnt) // 2 + cnt))
    t0 = time.perf_counter()
    rows = []
    with get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool, (out / "headroom.jsonl").open("w") as fh:
        for i, rs in enumerate(pool.imap_unordered(task, tasks), 1):
            for r in rs:
                fh.write(json.dumps(r) + "\n")
            rows.extend(rs)
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("\n| Patient | Precision | pcodec8 vs payload | low half best / raw | high half best / raw | "
          "split best vs payload | mantissa bit entropy (median of bits) | uniform mantissa bits (median) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for patient in args.patients:
        for prec in ("f64", "f32"):
            rr = [r for r in rows if r["patient"] == patient and r["precision"] == prec]
            pay = sum(r["payload_bytes"] for r in rr)
            half = sum(r["half_raw_bytes"] for r in rr)
            ent = [r["mantissa_bit_entropy"] for r in rr if r["mantissa_bit_entropy"] is not None]
            uni = [r["mantissa_uniform_bits"] for r in rr if r["mantissa_uniform_bits"] is not None]
            print(f"| {patient} | {prec} | {sum(r['pcodec8_bytes'] for r in rr) / pay:.3f} | "
                  f"{sum(r['lo_best_bytes'] for r in rr) / half:.3f} | {sum(r['hi_best_bytes'] for r in rr) / half:.3f} | "
                  f"{sum(r['lo_best_bytes'] + r['hi_best_bytes'] for r in rr) / pay:.3f} | "
                  f"{np.median(ent):.1f} of {rr[0]['mantissa_bits']} | {np.median(uni):.0f} |")


if __name__ == "__main__":
    main()
