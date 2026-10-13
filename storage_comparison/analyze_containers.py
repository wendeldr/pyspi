"""Summarise containers.jsonl runs into per-patient tables, corruption outcomes, and a cohort extrapolation.

Usage: analyze_containers.py [--common-metrics] reports/<run> [reports/<run2> ...]
--common-metrics keeps, per patient, only metrics measured for every variant of that patient
(needed when runs over different metric subsets are combined, e.g. a chunk-size sweep).
Writes containers_summary.json next to the first run and prints markdown tables.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "reports" / "inventory" / "cohort_inventory.json"
SUM_KEYS = ("bytes", "alloc", "files", "write_s", "fsync_s", "cold_read_s", "cold_mean_s", "warm_read_s",
            "window_read_s", "raw_f64_bytes", "load_s", "pack_s")


def load(runs):
    rows, corr = [], []
    for run in runs:
        for line in (Path(run) / "containers.jsonl").read_text().splitlines():
            r = json.loads(line)
            (rows if "metric" in r else corr).append(r)
    return rows, corr


SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")
_SRC_CACHE: dict = {}


def source_bytes(patient, metric):
    """Stored bytes of one source metric array (stat only; the source is never opened for writing)."""
    key = (patient, metric)
    if key not in _SRC_CACHE:
        d = SOURCE / f"{patient}.zarr" / "connectivity" / "0.5s" / metric
        _SRC_CACHE[key] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return _SRC_CACHE[key]


def main():
    common = "--common-metrics" in sys.argv
    runs = [a for a in sys.argv[1:] if a != "--common-metrics"]
    rows, corr = load(runs)
    if common:
        per = collections.defaultdict(lambda: collections.defaultdict(set))
        for r in rows:
            per[r["patient"]][r["variant"]].add(r["metric"])
        keep = {pt: set.intersection(*v.values()) for pt, v in per.items()}
        rows = [r for r in rows if r["metric"] in keep[r["patient"]]]
        print("common metrics per patient:", {pt: len(m) for pt, m in keep.items()})
    inv = {r["patient"].removesuffix(".zarr"): r for r in json.loads(INVENTORY.read_text())}
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        a = agg[(r["patient"], r["variant"])]
        a["metrics"] += 1
        a["source_bytes"] += source_bytes(r["patient"], r["metric"])
        for k in SUM_KEYS:
            a[k] += r[k]
    out = {}
    for (patient, variant), a in sorted(agg.items()):
        src = inv[patient]
        complete = a["metrics"] == src["metrics"]
        d = dict(a)
        d.update(patient=patient, variant=variant, complete=complete, n=src["n"], windows=src["windows"],
                 ratio_vs_raw=a["raw_f64_bytes"] / a["bytes"],
                 fraction_of_source=a["bytes"] / a["source_bytes"],
                 write_total_s=a["write_s"] + a["fsync_s"])
        out[f"{patient}|{variant}"] = d
    # markdown tables per patient
    for patient in sorted({k.split("|")[0] for k in out}):
        src = inv[patient]
        print(f"\n### Patient {patient} ({src['n']} channels, {src['windows']} windows; source {src['conn_bytes'] / 2**30:.2f} GiB, "
              f"{src['conn_files']:,} files)\n")
        print("| Variant | Metrics | GiB | vs source | vs raw | Files | Write+fsync s | Cold read s | Cold mean s | Warm read s | 1-window s (sum) |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        items = [v for k, v in out.items() if k.startswith(patient + "|")]
        items.sort(key=lambda v: (v["variant"].split("@")[1], v["bytes"]))
        for v in items:
            frac = f"{v['fraction_of_source']:.3f}" + ("" if v["complete"] else " (subset)")
            print(f"| {v['variant']} | {v['metrics']} | {v['bytes'] / 2**30:.3f} | {frac} | {v['ratio_vs_raw']:.2f}x | {v['files']:,} | "
                  f"{v['write_total_s']:.1f} | {v['cold_read_s']:.1f} | {v['cold_mean_s']:.1f} | {v['warm_read_s']:.1f} | {v['window_read_s']:.2f} |")
    # corruption outcomes
    if corr:
        print("\n### Corruption (one flipped byte per trial)\n")
        print("| Patient | Variant | Trial | Full read | Windows recoverable / error / silent |")
        print("| --- | --- | --- | --- | --- |")
        for c in corr:
            for t in c["results"]:
                print(f"| {c['patient']} | {c['corruption']} | {t['position']} | {t['full_read']} | "
                      f"{t['windows_recoverable']} / {t['windows_error']} / {t['windows_silent']} |")
    # cohort extrapolation: fraction of current source bytes as a function of channel count
    per_variant = collections.defaultdict(list)
    for v in out.values():
        if v["fraction_of_source"] is not None:
            per_variant[v["variant"]].append((v["n"], v["fraction_of_source"], v["files"] / v["metrics"]))
    proj = {}
    total_src = sum(r["conn_bytes"] for r in inv.values())
    for variant, pts in per_variant.items():
        pts.sort()
        ns, fr, fpm = zip(*pts)
        est = 0.0
        files = 0.0
        for r in inv.values():
            f = float(np.interp(r["n"], ns, fr))  # constant beyond measured range
            est += f * r["conn_bytes"]
            files += np.interp(r["n"], ns, fpm) * r["metrics"]
        proj[variant] = dict(measured_channel_counts=list(ns), estimated_bytes=est, estimated_fraction=est / total_src,
                             estimated_files=files)
    if proj:
        print(f"\n### Cohort extrapolation (NOT a measured conversion; source total {total_src / 2**30:.1f} GiB, "
              f"{sum(r['conn_files'] for r in inv.values()):,} files)\n")
        print("| Variant | Measured at n | Est. GiB | Est. fraction | Est. files |")
        print("| --- | --- | ---: | ---: | ---: |")
        for variant, p in sorted(proj.items(), key=lambda x: x[1]["estimated_bytes"]):
            print(f"| {variant} | {p['measured_channel_counts']} | {p['estimated_bytes'] / 2**30:.0f} | "
                  f"{p['estimated_fraction']:.3f} | {p['estimated_files']:,.0f} |")
    (Path(runs[0]) / "containers_summary.json").write_text(json.dumps(dict(variants=out, corruption=corr, projection=proj), indent=1))


if __name__ == "__main__":
    main()
